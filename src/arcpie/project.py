from __future__ import annotations

from pathlib import Path
import re
from collections.abc import (
    Sequence,
    Iterator,
)
from collections import UserString
from typing import (
    Literal,
    TypeVar,
    Generic,
    Any,
    overload,
)

from functools import cached_property

from arcpy.mp import ArcGISProject

from arcpy._mp import (
    Map as _Map,
    Layout as _Layout,
    Layer as _Layer,
    Table as _Table,
    Report as _Report,
)

_T = TypeVar('_T')
_Default = TypeVar('_Default')

# String Wrapper to make wildcards clear
# Since the return type of a wildcard index is different from a string index
class Wildcard(UserString): ...
def wildcard(wc: str) -> Wildcard:
    return Wildcard(wc)

class _Wrapper(Generic[_T]):
    """Internal wrapper class for wrapping existing objects with new functionality"""
    def __init__(self, obj: _T, parent: _MappingObject|Project|None=None) -> None:
        self._obj = obj
        self._parent = parent
    
    def __getattr__(self, attr: str) -> Any:
        return getattr(self._obj, attr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{getattr(self, 'name')}', parent={getattr(self, '_parent')})"

# Wrappers around existing Mapping Objects to allow extensible functionality
class Map(_Map, _Wrapper[_Map]):
    @property
    def layers(self) -> LayerManager:
        return LayerManager([Layer(l, self) for l in self._obj.listLayers()])
    
class Layer(_Wrapper[_Layer], _Layer): ...
class Layout(_Wrapper[_Layout], _Layout): ...
class Table(_Wrapper[_Table], _Table): ...
class Report(_Wrapper[_Report], _Report): ...

_MappingObject = TypeVar('_MappingObject', Map, Layout, Layer, Table, Report)
def name_of(o: _MappingObject, skip_uri: bool=False, uri_only: bool=False) -> str:
    """Handle the naming hierarchy of mapping objects URI -> longName -> name
    
    Allow setting flags to get specific names
    """
    _uri: str|None = getattr(o, 'URI', None) if not skip_uri else None
    _long_name: str|None = getattr(o, 'longName', None) # longName will identify Grouped Layers
    _name: str|None = getattr(o, 'name', None)
    _id: str = str(id(o)) # Fallback to a locally unique id (should never happen)
    if uri_only:
        return _uri or f"{_name}: NO URI"
    return _uri or _long_name or _name or _id

class Manager(Generic[_MappingObject]):
    """Base access interfaces for all manager classes. Specific interfaces are defined in the subclass
    
    Index itentifiers are URI -> longName -> name depending on what is available in the managed class
    """
    
    def __init__(self, objs: Sequence[_MappingObject]) -> None:
        self._objects: dict[str, _MappingObject] = {
            name_of(o): o 
            for o in objs
        }
        
    @property
    def objects(self) -> list[_MappingObject]:
        return list(self._objects.values())
    
    @property
    def names(self) -> list[str]:
        # Skip the URI 
        return [name_of(o, skip_uri=True) for o in self.objects]
    
    @property
    def uris(self) -> list[str]:
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
                return self.objects[name]
            case re.Pattern():
                # Handle regex
                return [o for o in self.objects if name.match(name_of(o, skip_uri=True))]
            case Wildcard():
                # Handle wildcard
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

    def get(self, name: str, default: _Default=None) -> _MappingObject|list[_MappingObject]|_Default:
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name: str|_MappingObject) -> bool:
        match name:
            case str():
                return True if self.get(name) else False
            case _:
                return any(o == name for o in self.objects)

    def __iter__(self) -> Iterator[_MappingObject]:
        yield from (o for o in self._objects.values())

    def __len__(self) -> int:
        return len(self._objects)

class ReportManager(Manager[Report]): ...
class MapManager(Manager[Map]): ...
class LayerManager(Manager[Layer]): ...
class LayoutManager(Manager[Layout]): ...
class TableManager(Manager[Table]): ...

class Project(_Wrapper[ArcGISProject], ArcGISProject):
    def __init__(self, aprx_path: str|Path|Literal['CURRRENT']='CURRENT') -> None:
        self._path = str(aprx_path)
    
    def __repr__(self) -> str:
        return f"{Path(self.filePath).stem}"
    
    @cached_property
    def aprx(self) -> ArcGISProject:
        return ArcGISProject(self._path)
    
    @cached_property
    def maps(self) -> MapManager:
        return MapManager([Map(m, self) for m in self.aprx.listMaps()])
    
    @cached_property
    def layouts(self) -> LayoutManager:
        return LayoutManager([Layout(l, self) for l in self.aprx.listLayouts()])
    
    @cached_property
    def reports(self) -> ReportManager:
        return ReportManager([Report(r, self) for r in self.aprx.listReports()])
    
    @cached_property
    def broken_layers(self) -> LayerManager:
        return LayerManager([Layer(l, self) for l in self.aprx.listBrokenDataSources() if isinstance(l, _Layer)]) 
    
    @cached_property
    def broken_tables(self) -> TableManager:
        return TableManager([Table(t, self) for t in self.aprx.listBrokenDataSources() if isinstance(t, _Table)])
    
    def refresh(self, *, managers: Sequence[str]|None=None) -> None:
        """Clear cached object managers"""
        for prop in list(self.__dict__):
            if prop.startswith('_'):
                continue # Skip private instance attributes
            self.__dict__.pop(prop, None)