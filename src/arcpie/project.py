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

from arcpy.mp import ArcGISProject, LayerFile
from arcpy.cim.cimloader.cimtojson import CimJsonEncoder
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
from arcpy.cim import CIMStandaloneTable
from arcpy.cim import CIMMapDocument
from arcpy.cim.CIMCore import CIMDefinition

# Common layer CIM types 
CIMLayerType = (
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
    MapSeriesPDFSetting,
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
class Wildcard(UserString): 
    """Clarify that the string passed to a Manager index is a wildcard so the type checker knows you're getting a Sequence back"""
    pass

class MappingWrapper(Generic[_T]):
    """Internal wrapper class for wrapping existing objects with new functionality"""
    def __init__(self, obj: _T, parent: _MappingObject|Project|None=None) -> None:
        self._obj = obj
        self._parent = parent
            
    @property
    def parent(self):
        """The parent object for the wrapper
        
        `Project -> Map -> Layer`\n
        `Project -> Layout -> Map -> MapSeries`\n
        `Project -> Report`
        
        The general parent/child relationships are based on how you would access the object in ArcPro.
        Projects have maps, maps have layers, layouts have mapseries etc.
        """
        return self._parent
    
    @property
    def cim(self) -> CIMDefinition | None:
        try:
            return self._obj.getDefinition('V3')  #type: ignore
        except json.JSONDecodeError:
            return None
        
    @property
    def cim_dict(self) -> dict[str, Any] | None:
        if _cim := self.cim:
            return json.loads(json.dumps(self.cim, cls=CimJsonEncoder))
    
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
    def lyrx(self) -> dict[str, Any] | None:
        """Get a dictionary representation of the layer that can be saved to an lyrx file using `json.dumps`
        
        Note:
            GroupLayer objects will return a lyrx template with all sub-layers included
        """
        _def = self.cim_dict
        if _def is None:
            return None
        _lyrx: dict[str, Any] = { # Base required keys for lyrx file
            'type': 'CIMLayerDocument',
            'layers': [self.URI],
            'layerDefinitions': [_def],
        }
        
        # Handle Group Layers
        if self.isGroupLayer:
            _child_tables: list[str] = [uri for uri in _def.get('tables', [])]
            _child_layers: list[str] = [uri for uri in _def.get('layers', [])]
            if self.parent and isinstance(self.parent, Map) and _child_tables:
                _lyrx['tableDefinitions'] = [self.parent.tables.get(uri).lyrx for uri in _child_tables]
                _lyrx['layerDefinitions'] = [_def] + [self.parent.layers.get(uri).lyrx for uri in _child_layers]
                
        return _lyrx

    @property
    def cim(self) -> CIMLayerType: # pyright: ignore[reportIncompatibleMethodOverride]
        return super().cim # pyright: ignore[reportReturnType]
    
    def export_lyrx(self, out_dir: Path|str) -> None:
        """Export the layer to a lyrx file in the target directory
        
        Args:
            out_dir (Path|str): The location to export the mapx to
        """
        out_dir = Path(out_dir)
        target = (out_dir / self.longName)
        # Make Containing directory for grouped layers
        target.parent.mkdir(exist_ok=True, parents=True)
        target.with_suffix('.lyrx').write_text(json.dumps(self.lyrx))
    
    def import_lyrx(self, lyrx: Path|str) -> None:
        """Import the layer state from an lyrx file
        
        Args:
            lyrx (Path|str): The lyrx file to update this layer with
        """
        _lyrx = LayerFile(str(lyrx))
        _lyrx_layers = {l.name: l for l in _lyrx.listLayers()}
        for layer in ([self] + self.listLayers() if self.isGroupLayer else [self]):
            _lyrx_layer = _lyrx_layers.get(layer.name)
            if not _lyrx_layer:
                print(f'{self.name} not found in {str(lyrx)}')
                continue
            # Update Connection
            _lyrx_layer.updateConnectionProperties(None, layer.connectionProperties) # type: ignore
            _lyrx_layer_cim = _lyrx_layer.getDefinition('V3')
            self.setDefinition(_lyrx_layer_cim)
    
class Bookmark(MappingWrapper[_Bookmark], _Bookmark): ...

class BookmarkMapSeries(MappingWrapper[_BookmarkMapSeries], _BookmarkMapSeries):
    """Wrapper around an arcpy.mp BookmarkMapSeries object that provides an ergonomic interface"""
    
    def __iter__(self) -> Iterator[BookmarkMapSeries]:
        _orig_page = self.currentPageNumber
        for page in range(1, self.pageCount):
            self.currentPageNumber = page
            yield self
        if _orig_page:
            self.currentPageNumber = _orig_page
    
    def __getitem__(self, page: int|str|_Bookmark) -> BookmarkMapSeries:
        """Allow indexing the BookmarkMapSeries by name, index, or Bookmark object"""
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
    """Wrapper around an arcpy.mp MapSeries object that provides an ergonomic interface"""
    @property
    def layer(self) -> Layer:
        """Get the mapseries target layer"""
        return Layer(self.indexLayer, self.map)
    
    @property # Passthrough
    def feature_class(self) -> FeatureClass[Any]:
        """Get the FeatureClass of the parent layer"""
        return self.layer.feature_class
    
    @property
    def map(self) -> Map:
        """Get the map object that is being seriesed"""
        return Map(self.mapFrame.map, self.parent.parent) #type: ignore (if a MapSeries is initialized this will be a map)
    
    @property
    def pageRow(self): # type: ignore (prevent access to uninitialized pageRow raising RuntimeError)
        """Get a Row object for the active mapseries page"""
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
        """Get all valid page names for the MapSeries"""
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
        """Get the name of the active mapseries page"""
        return self.page_values.get(self.page_field, 'No Page')
    
    def to_pdf(self, **settings: Unpack[MapSeriesPDFSetting]) -> bytes:
        """Export the MapSeries to a PDF, See Layer.to_pdf for more info
        
        Args:
            **settings (Unpack[MapSeriesPDFSetting]): Passthrough kwargs for layout.exportToPDF
        
        Note:
            By default, printing a mapseries will print all pages to a single file. To only print
            the active page:
            ```python
            >>> ms.to_pdf(page_range_type='CURRENT')
            ```
        """
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
        """Allow indexing a mapseries by a page name or a page index/number"""
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
        """Get a LayerManager for all layers in the Map"""
        return LayerManager(Layer(l, self) for l in self.listLayers() or [])

    @property
    def tables(self) -> TableManager:
        """Get a TableManager for all tables in the Map"""
        return TableManager(Table(t, self) for t in self.listTables() or [])

    @property
    def bookmarks(self) -> BookmarkManager:
        """Get a BookmarkManager for all bookmarks in the Map"""
        return BookmarkManager(Bookmark(b, self) for b in self.listBookmarks() or [])
    
    @property
    def elevation_surfaces(self) -> ElevationSurfaceManager:
        """Get an ElevationSurfaceManager for all elevation surfaces in the Map"""
        return ElevationSurfaceManager(ElevationSurface(es, self) for es in self.listElevationSurfaces() or [])
    
    @property
    def mapx(self) -> dict[str, Any]:
        with NamedTemporaryFile(suffix='.mapx') as tmp:
            self.exportToMAPX(tmp.name)
            return json.loads(Path(tmp.name).read_text(encoding='utf-8'))
    
    @property
    def cim(self) -> CIMMapDocument: # pyright: ignore[reportIncompatibleMethodOverride]
        return self.getDefinition('V3')  # pyright: ignore[reportReturnType]
    
    @property
    def cim_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.cim, cls=CimJsonEncoder))
    
    def export_mapx(self, out_dir: Path|str) -> None:
        """Export the map definitions to a mapx file in the target directory
        
        Args:
            out_dir (Path|str): The location to export the mapx to
        """
        out_dir = Path(out_dir)
        (out_dir / self.name).write_text(json.dumps(self.mapx))
    
    def export_assoc_lyrx(self, out_dir: Path|str) -> None:
        """Export all child layers to lyrx files the target directory
        
        Args:
            out_dir (Path|str): The location to export the lyrx files to
        """
        out_dir = Path(out_dir)
        for layer in self.layers:
            layer.export_lyrx(out_dir)
    
    def import_assoc_lyrx(self, lyrx_dir: Path|str) -> None:
        lyrx_dir = Path(lyrx_dir)
        for lyrx_path in lyrx_dir.rglob('*.lyrx'):
            _lyrx_name = str(lyrx_path.relative_to(lyrx_dir).with_suffix(''))
            if _lyrx_name in self.layers:
                self.layers[_lyrx_name].import_lyrx(lyrx_path)
            elif _lyrx_name in self.tables:
                pass # Not implemented yet
    
    def import_mapx(self, mapx: Path|str) -> None:
        raise NotImplementedError()
    
class Layout(MappingWrapper[_Layout], _Layout):
    
    @property
    def mapseries(self) -> MapSeries | BookmarkMapSeries| None:
        """Get the Layout MapSeries/BookmarkMapSeries if it exists"""
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
        """Get an `arcpie.Table` object from the TableLayer"""
        return DataTable.from_table(self)

    @property
    def cim(self) -> CIMStandaloneTable:
        return super().cim # pyright: ignore[reportReturnType]

    @property
    def lyrx(self) -> dict[str, Any]:
        _def = self.cim_dict
        _lyrx: dict[str, Any] = { # Base required keys for lyrx file
            'type': 'CIMLayerDocument',
            'tables': [self.URI],
            'tableDefinitions': [_def],
        }
        return _lyrx

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
    """Wrapper for an ArcGISProject (.aprx)
    
    Usage:
        ```python
        >>> prj = Project('<path/to/aprx>')
        >>> lay = prj.layouts.get('My Layout')
        >>> Path('My Layout.pdf').write_bytes(prj.layouts.get('My Layout').to_pdf())
        4593490 # Bytes written
        >>> for map in prj.maps:
        ...     print(f'{map.name} has {len(map.layers)} layers')
        My Map has 5 layers
        My Map 2 has 15 layers
        Other Map has 56 layers
        ```
    """
    def __init__(self, aprx_path: str|Path|Literal['CURRRENT']='CURRENT') -> None:
        self._path = str(aprx_path)
    
    def __repr__(self) -> str:
        return f"Project({Path(self.aprx.filePath).stem}.aprx)"
    
    @property
    def name(self) -> str:
        """Get the file name of the wrapped aprx minus the file extension"""
        return Path(self.aprx.filePath).stem
    
    @property
    def aprx(self) -> ArcGISProject:
        """Get the base ArcGISProject for the Project"""
        return ArcGISProject(self._path)
    
    @property
    def maps(self) -> MapManager:
        """Get a MapManager for the Project maps"""
        return MapManager(Map(m, self) for m in self.aprx.listMaps())
    
    @property
    def layouts(self) -> LayoutManager:
        """Get a LayoutManager for the Project layouts"""
        return LayoutManager(Layout(l, self) for l in self.aprx.listLayouts())
    
    @property
    def reports(self) -> ReportManager:
        """Get a ReportManager for the Project reports"""
        return ReportManager(Report(r, self) for r in self.aprx.listReports())
    
    @property
    def broken_layers(self) -> LayerManager:
        """Get a LayerManager for all layers in the project with broken datasources"""
        return LayerManager(Layer(l, self) for l in self.aprx.listBrokenDataSources() if isinstance(l, _Layer))
    
    @property
    def broken_tables(self) -> TableManager:
        """Get a TableManager for all tables in the project with broken datasources"""
        return TableManager(Table(t, self) for t in self.aprx.listBrokenDataSources() if isinstance(t, _Table))
    
    @overload
    def __getitem__(self, name: str) -> Map | Layout | Report: ...
    @overload
    def __getitem__(self, name: Wildcard) -> list[Map] | list[Layout] | list[Report]: ...
    def __getitem__(self, name: str | Wildcard) -> Any | list[Any]:
        """Resolve the name by looking in Maps, then Layouts, then Reports"""
        _obj = self.maps.get(name, None) or self.layouts.get(name, None) or self.reports.get(name, None)
        if _obj is None:
            raise KeyError(f'{name} not found in {self.name}')
        return _obj
    
    @overload
    def get(self, name: str, default: _Default) -> Map | Layout | Report | _Default: ...
    @overload
    def get(self, name: Wildcard, default: _Default) -> list[Map] | list[Layout] | list[Report] | _Default: ...
    def get(self, name: str | Wildcard, default: _Default=None) -> Any | _Default:
        try:
            return self[name]
        except KeyError:
            return default
    
    def save(self) -> None:
        """Save this project"""
        self.aprx.save()
    
    def save_as(self, path: Path|str) -> Project:
        """Saves the project under a new name
        
        Args:
            path (Path|str): The filepath of the new aprx
        
        Returns:
            (Project): A Project representing the new project file
        
        Note:
            Saving a Project as a new project will not update this instance, and instead returns a new
            Project instance targeted at the new file
        """
        path = Path(path)
        self.aprx.saveACopy(str(path.with_suffix('.aprx')))
        return Project(path)
    
    def import_pagx(self, pagx: Path|str, *, reuse_existing_maps: bool=True) -> Layout:
        """Import a pagx file into this project
        
        Args:
            pagx (Path|str): The path to the pagx document
            
        Returns:
            (Layout): A Layout parented to this project
        """
        pagx = Path(pagx)
        if not pagx.suffix == '.pagx':
            raise ValueError(f'{pagx} is not a pagx file!')
        
        _imported = self.aprx.importDocument(str(pagx), reuse_existing_maps=reuse_existing_maps)
        # Ensure that the imported document is a Layout
        if not isinstance(_imported, _Layout):
            self.aprx.deleteItem(_imported)
            raise ValueError(f'{pagx} is not a valid pagx file!')
        return Layout(_imported, parent=self)
    
    def export_pagx(self, target_dir: Path|str) -> None:
        """Export all layouts to a directory"""
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
        for layout in self.layouts:
            (target_dir / layout.name).with_suffix('.pagx').write_text(json.dumps(layout.pagx))
    
    def import_mapx(self, mapx: Path|str) -> Map:
        """Import a mapx file into this project
        
        Args:
            mapx (Path|str): The path to the mapx document
            
        Returns:
            (Map): A Map parented to this project
        """
        mapx = Path(mapx)
        if not mapx.suffix == '.mapx':
            raise ValueError(f'{mapx} is not a mapx file!')
        
        _imported = self.aprx.importDocument(str(mapx))
        # Ensure that the imported document is a Map
        if not isinstance(_imported, _Map):
            self.aprx.deleteItem(_imported)
            raise ValueError(f'{mapx} is not a valid mapx file!')
        return Map(_imported, parent=self)
    
    def export_mapx(self, target_dir: Path|str) -> None:
        """Export all maps to a directory"""
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
        for m in self.maps:
            (target_dir / m.name).with_suffix('.mapx').write_text(json.dumps(m.mapx))
    
    def export_layers(self, target_dir: Path|str, *, skip_groups: bool = False, skip_grouped: bool = False) -> None:
        """Export all layers in the project to a structured directory of layerfiles
        
        Args:
            target_dir (Path|str): The target directory to export the layerfiles to
            skip_groups (bool): Skip group layerfiles and export each layer individually in a group subdirectory (default: False)
            skip_grouped (bool): Inverse of skip groups and instead only exports the group lyrx, skipping the individual layers (default: False)
        """
        target_dir = Path(target_dir)
        for m in self.maps:
            m.export_assoc_lyrx(target_dir, skip_groups=skip_groups, skip_grouped=skip_grouped)
    
    def import_layers(self, src_dir: Path|str) -> None:
        """Import a structured directory of layerfiles generated with `export_layers`
        
        Args:
            src_dir (Path|str): A directory containing layer files in map directories and group directories

        Note:
            CIM changes require the APRX to be saved to take effect. If you are accessing this
            layer via a Project, use `project.save()` after importing the layerfile
        """
        src_dir = Path(src_dir)
        for m in self.maps:
            map_dir = src_dir / m.name
            if not map_dir.exists():
                print(f'Map {m.name} does not have a valid source in the source directory, skipping')
                continue
            m.import_assoc_lyrx(map_dir)
            