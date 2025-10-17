from __future__ import annotations

from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from collections.abc import (
    Sequence,
    Iterator,
    Iterable,
)
from collections import UserString
from typing import (
    Literal,
    TypeVar,
    Generic,
    Any,
    Unpack,
    overload,
)

from functools import cached_property
import json

from arcpy.mp import ArcGISProject
from arcpy.cim.cimloader import CimJsonEncoder
from arcpy.cim.CIMVectorLayers import (
    CIMAnnotationLayer,
    CIMFeatureLayer,
    CIMVectorTileLayer,
)
from arcpy.cim.CIMServiceLayers import (
    CIMTiledServiceLayer,
)
from arcpy.cim.CIMLayer import (
    CIMGraphicsLayer,
    CIMGroupLayer,
)

# Common layer CIM types 
LayerCimTypes = (
    CIMFeatureLayer | CIMVectorTileLayer | CIMAnnotationLayer | 
    CIMTiledServiceLayer | CIMGraphicsLayer | CIMGroupLayer
)

# Shadow all _mp types with interface Wrappers
from arcpy._mp import (
    Map as _Map,
    Layout as _Layout,
    Layer as _Layer,
    Table as _Table,
    Report as _Report,
    MapSeries as _MapSeries,
    BookmarkMapSeries as _BookmarkMapSeries,
    Bookmark as _Bookmark,
    ElevationSurface as _ElevationSurface,
)

from arcpie._types import (
    PDFSetting,
    MapseriesPDFSetting,
    MapseriesPDFDefault,
    PDFDefault,
)

from arcpie.featureclass import (
    FeatureClass, 
    Table as DataTable, # Alias Table to prevent conflict with Mapping Table
)

_T = TypeVar('_T')
_Default = TypeVar('_Default')

# String Wrapper to make wildcards clear
# Since the return type of a wildcard index is different from a string index
class Wildcard(UserString): ...

class MappingWrapper(Generic[_T]):
    """Internal wrapper class for wrapping existing objects with new functionality"""
    def __init__(self, obj: _T, parent: _MappingObject|Project|None=None) -> None:
        self._obj = obj
        self._parent = parent
    
    @property
    def parent(self):
        return self._parent
    
    def __getattr__(self, attr: str) -> Any:
        return getattr(self._obj, attr)

    def __repr__(self) -> str:
        return f"{self._obj.__class__.__name__}({name_of(self._obj, skip_uri=True)})"
    
    def __eq__(self, other: MappingWrapper[Any] | object) -> bool:
        if hasattr(other, '_obj'):
            return self._obj is getattr(other, '_obj', None)
        else:
            return super().__eq__(other)

# Wrappers around existing Mapping Objects to allow extensible functionality 
class Layer(MappingWrapper[_Layer], _Layer):
    @property
    def feature_class(self) -> FeatureClass[Any]:
        """Get a `arcpie.FeatureClass` object that is initialized using the layer and its current state"""
        return FeatureClass[Any].from_layer(self)
    
    @property
    def symbology(self) -> Any:
        """Get the base symbology object for the layer"""
        return self._obj.symbology
    
    @property
    def lyrx(self) -> dict[str, Any]:
        """Get a dictionary representation of the layer that can be saved to an lyrx file using `json.dumps`"""
        if self.isWebLayer:
            return {} # Web layers have no valid CIM
        _def = json.loads(json.dumps(self.cim, cls=CimJsonEncoder))
        if self.isGroupLayer and isinstance(self.parent, Map) and isinstance(self.cim, CIMGroupLayer):
            _children: list[Layer] = [self.parent.layers.get(uri) for uri in self.cim_dict['layers']]
            _lyrx: dict[str, Any] = {
                'type': 'CIMLayerDocument',
                'layers': [self.URI],
                'layerDefinitions': [self.cim_dict] + [child.cim_dict for child in _children]
            }
        elif self.isFeatureLayer:
            _lyrx: dict[str, Any] = { # Base required keys for lyrx file
                'type': 'CIMLayerDocument',
                'layers': [self.URI],
                'layerDefinitions': [_def],
            }
        else:
            _lyrx = {}
        return _lyrx

    @property
    def cim(self) -> LayerCimTypes:
        """Get the raw CIM V3 definition of the Layer, see `arcpy.cim` for more info"""
        return self.getDefinition('V3')  # pyright: ignore[reportReturnType]
    
    @property
    def cim_dict(self) -> dict[str, Any]:
        """Get a dictionary representation of the layer CIM instead of a CIM object"""
        return json.loads(json.dumps(self.cim, cls=CimJsonEncoder))
class Bookmark(MappingWrapper[_Bookmark], _Bookmark): ...

class BookmarkMapSeries(MappingWrapper[_BookmarkMapSeries], _BookmarkMapSeries):
    
    def __iter__(self) -> Iterator[BookmarkMapSeries]:
        _orig_page = self.currentPageNumber
        for page in range(1, self.pageCount):
            self.currentPageNumber = page
            yield self
        if _orig_page:
            self.currentPageNumber = _orig_page
    
    def __getitem__(self, page: int|str|_Bookmark) -> BookmarkMapSeries:
        match page:
            case _Bookmark():
                self.currentBookmark = page
            case str():
                self.currentPageNumber = self.getPageNumberFromName(page)
            case int():
                self.currentPageNumber = page
        return self

    def __len__(self) -> int:
        return len(self.bookmarks)

class MapSeries(MappingWrapper[_MapSeries], _MapSeries):
    
    @property
    def layer(self) -> Layer:
        """Get the mapseries target layer"""
        return Layer(self.indexLayer, self.map)
    
    @property # Passthrough
    def feature_class(self) -> FeatureClass[Any]:
        return self.layer.feature_class
    
    @property
    def map(self) -> Map:
        """Get the map object that is being seriesed"""
        return Map(self.mapFrame.map, self.parent.parent) #type: ignore (if a MapSeries is initialized this will be a map)
    
    @property
    def pageRow(self): # type: ignore (prevent access to uninitialized pageRow raising RuntimeError)
        try:
            return self._obj.pageRow
        except RuntimeError:
            return None
    
    @property
    def page_field(self) -> str:
        """Get fieldname used as pagename"""
        return self.pageNameField.name
    
    @cached_property
    def page_field_names(self) -> list[str]:
        """Get all fieldnames for the mapseriesed features"""
        return [f for f in self.feature_class.fields if not f.startswith('@')]
    
    @property
    def valid_pages(self) -> list[str]:
        return list(self.feature_class[self.page_field])
    
    @property
    def page_values(self) -> dict[str, Any]:
        """Get a mapping of values for the current page"""
        if not self.pageRow:
            return {} # pageRow is unset with no active page
        
        # Need to access a private `_asdict` method of Row because getValue is broken
        return {f: self.pageRow._asdict().get(f) for f in self.page_field_names}
    
    @property
    def current_page_name(self) -> str:
        return self.page_values.get(self.page_field, 'No Page')
    
    def to_pdf(self, **settings: Unpack[MapseriesPDFSetting]) -> bytes:
        _settings = MapseriesPDFDefault.copy()
        _settings.update(settings)
        with NamedTemporaryFile() as tmp:
            return Path(self.exportToPDF(tmp.name, **_settings)).read_bytes()
    
    def __iter__(self) -> Iterator[MapSeries]:
        _orig_page = self.currentPageNumber
        for page in range(1, self.pageCount):
            self.currentPageNumber = page
            yield self
        if _orig_page:
            self.currentPageNumber = _orig_page
    
    def __getitem__(self, page: int|str) -> MapSeries:
        match page:
            case str():
                if page not in self.valid_pages:
                    raise KeyError(f"{page} is not a valid page name!")
                self.currentPageNumber = self.getPageNumberFromName(page)
            case int():
                if page not in range(1, self.pageCount):
                    raise IndexError(f"{self} only has {self.pageCount} pages, {page} out of range")
                self.currentPageNumber = page
        return self

    def __len__(self) -> int:
        return self.pageCount

    def __repr__(self) -> str:
        return f'MapSeries<{self.layer.name} @ {self.current_page_name}>'

class ElevationSurface(MappingWrapper[_ElevationSurface], _ElevationSurface): ...

class Map(_Map, MappingWrapper[_Map]):
    @property
    def layers(self) -> LayerManager:
        return LayerManager(Layer(l, self) for l in self.listLayers())

    @property
    def tables(self) -> TableManager:
        return TableManager(Table(t, self) for t in self.listTables())

    @property
    def bookmarks(self) -> BookmarkManager:
        return BookmarkManager(Bookmark(b, self) for b in self.listBookmarks())
    
    @property
    def elevation_surfaces(self) -> ElevationSurfaceManager:
        return ElevationSurfaceManager(ElevationSurface(es, self) for es in self.listElevationSurfaces())
    
class Layout(MappingWrapper[_Layout], _Layout):
    
    @property
    def mapseries(self) -> MapSeries | BookmarkMapSeries| None:
        if not self.mapSeries:
            return None
        if isinstance(self.mapSeries, _MapSeries):
            return MapSeries(self.mapSeries, self)
        return BookmarkMapSeries(self.mapSeries, self)
    
    def to_pdf(self, **settings: Unpack[PDFSetting]) -> bytes:
        """Get the bytes for a pdf export of the Layout
        
        Args:
            **settings (PDFSetting): Optional settings for the export (default: `PDFDefault`)
        
        Returns:
            (BufferedReader): Byte stream of the printed PDF
        
        Example:
            ```python
                # Get the layout object from a project file
                lyt = prj.layouts['Layout_1']
                
                # Create a pdf then write the output of to_pdf to it
                pdf = Path('pdf_path').write_bytes(lyt.to_pdf())
            ```   
        """
        with NamedTemporaryFile() as tmp:
            _settings = PDFDefault.copy()
            for arg in _settings:
                if val := settings.get(arg):
                    _settings[arg] = val
            pdf = self.exportToPDF(tmp.name, **_settings)
            return Path(pdf).read_bytes()

class Table(MappingWrapper[_Table], _Table):
    @property
    def table(self) -> DataTable:
        return DataTable.from_table(self)

class Report(MappingWrapper[_Report], _Report): ...

# Unified typevar for valid wrapped mapping objects
_MappingObject = TypeVar(
    '_MappingObject', 
    Layer, 
    Bookmark, 
    BookmarkMapSeries, 
    MapSeries, 
    ElevationSurface, 
    Map, 
    Layout, 
    Table, 
    Report,
)

def name_of(o: Any, skip_uri: bool=False, uri_only: bool=False) -> str:
    """Handle the naming hierarchy of mapping objects URI -> longName -> name
    
    Allow setting flags to get specific names
    
    Note:
        If a URI is requested and no `URI` attribute is available in object, 
        `'obj.name: NO URI(id(obj))'` will be returned, e.g. `'my_bookmark: NO URI(1239012093)'`
    """
    _uri: str|None = getattr(o, 'URI', None) if not skip_uri else None
    _long_name: str|None = getattr(o, 'longName', None) # longName will identify Grouped Layers
    _name: str|None = getattr(o, 'name', None)
    _id: str = str(id(o)) # Fallback to a locally unique id (should never happen)
    if uri_only:
        return _uri or f"{_name}: NO URI({id(o)})"
    return _uri or _long_name or _name or _id

class Manager(Generic[_MappingObject]):
    """Base access interfaces for all manager classes. Specific interfaces are defined in the subclass
    
    Index itentifiers are URI -> longName -> name depending on what is available in the managed class
    """
    
    def __init__(self, objs: Iterable[_MappingObject]) -> None:
        self._objects: dict[str, _MappingObject] = {
            name_of(o): o 
            for o in objs
        }
        
    @property
    def objects(self) -> list[_MappingObject]:
        """Get a list of all managed objects"""
        return list(self._objects.values())
    
    @property
    def names(self) -> list[str]:
        """Get the names of all managed objects (skips URIs)"""
        return [name_of(o, skip_uri=True) for o in self.objects]
    
    @property
    def uris(self) -> list[str]:
        """Get URIs/CIMPATH for all managed objects
        
        Note:
            Default to a Python id() call if no URI is present
        """
        return [name_of(o, uri_only=True) for o in self.objects]
    
    @overload
    def __getitem__(self, name: str) -> _MappingObject: ...
    @overload
    def __getitem__(self, name: int) -> _MappingObject: ...
    @overload
    def __getitem__(self, name: Wildcard) -> list[_MappingObject]: ...
    @overload
    def __getitem__(self, name: re.Pattern[str]) -> list[_MappingObject]: ...
    @overload
    def __getitem__(self, name: slice) -> list[_MappingObject]: ...
    
    def __getitem__(self, name: str|Wildcard|re.Pattern[str]|int|slice) -> _MappingObject|list[_MappingObject]:
        """Access objects using a regex pattern (re.compile), wildcard (*STRING*), index, slice, name, or URI"""
        if isinstance(name, str) and '*' in name:
            name = Wildcard(name) # Allow passing a wildcard directly (lose type inference)
        match name:
            case int() | slice():
                return self.objects[name] # Will raise IndexError
            case re.Pattern():
                return [o for o in self.objects if name.match(name_of(o, skip_uri=True))]
            case Wildcard():
                return [o for o in self.objects if all(part in name_of(o, skip_uri=True) for part in name.split('*'))]
            case str(name) if name in self._objects:
                return self._objects[name]
            case str(name) if name in self.names:
                for o in self.objects:
                    if name_of(o, skip_uri=True) == name:
                        return o
            case _ :
                pass # Fallthrough to raise a KeyError
            
        raise KeyError(f'{name} not found in objects: ({self.names})')

    @overload
    def get(self, name: str) -> _MappingObject: ...
    @overload
    def get(self, name: Wildcard) -> list[_MappingObject]: ...
    @overload
    def get(self, name: str, default: _Default) -> _MappingObject | _Default: ...
    @overload
    def get(self, name: Wildcard, default: _Default) -> list[_MappingObject] | _Default: ...
    def get(self, name: str|Wildcard, default: _Default|None=None) -> _MappingObject|list[_MappingObject]|_Default|None:
        """Get a value from the Project with a safe default value"""
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name: str|_MappingObject) -> bool:
        """Check to see if a URI/name is present in the Manager"""
        match name:
            case str():
                return True if self.get(name) else False
            case _:
                return any(o == name for o in self.objects)

    def __iter__(self) -> Iterator[_MappingObject]:
        return iter(self._objects.values())

    def __len__(self) -> int:
        return len(self._objects)

# Inherit Manager to allow extension if needed
class ReportManager(Manager[Report]): ...
class MapManager(Manager[Map]): ...
class LayerManager(Manager[Layer]): ...
class LayoutManager(Manager[Layout]): ...
class TableManager(Manager[Table]): ...
class BookmarkManager(Manager[Bookmark]): ...
class ElevationSurfaceManager(Manager[ElevationSurface]): ...

class Project:
    def __init__(self, aprx_path: str|Path|Literal['CURRRENT']='CURRENT') -> None:
        self._path = str(aprx_path)
    
    def __repr__(self) -> str:
        return f"Project({Path(self.aprx.filePath).stem}.aprx)"
    
    @cached_property
    def name(self) -> str:
        return Path(self.aprx.filePath).stem
    
    @cached_property
    def aprx(self) -> ArcGISProject:
        return ArcGISProject(self._path)
    
    @cached_property
    def maps(self) -> MapManager:
        return MapManager(Map(m, self) for m in self.aprx.listMaps())
    
    @cached_property
    def layouts(self) -> LayoutManager:
        return LayoutManager(Layout(l, self) for l in self.aprx.listLayouts())
    
    @cached_property
    def reports(self) -> ReportManager:
        return ReportManager(Report(r, self) for r in self.aprx.listReports())
    
    @cached_property
    def broken_layers(self) -> LayerManager:
        return LayerManager(Layer(l, self) for l in self.aprx.listBrokenDataSources() if isinstance(l, _Layer))
    
    @cached_property
    def broken_tables(self) -> TableManager:
        return TableManager(Table(t, self) for t in self.aprx.listBrokenDataSources() if isinstance(t, _Table))
    
    def refresh(self, *, managers: Sequence[str]|None=None) -> None:
        """Clear cached object managers
        
        Args:
            managers (Sequence[str]|None): Optionally limit cache clearing to certain managers (attribute name)
        """
        for prop in list(self.__dict__):
            if prop.startswith('_') or managers and prop not in managers:
                continue # Skip private instance attributes and non-requested
            self.__dict__.pop(prop, None)