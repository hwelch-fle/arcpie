from __future__ import annotations

from arcpy import Parameter
from arcpy.mp import ArcGISProject
from abc import ABC
from typing import Any, overload, SupportsIndex

class ToolboxABC(ABC):
    def __init__(self) -> None:
        self.label: str
        self.alias: str
        self.tools: list[type[ToolABC]]

class ToolABC(ABC):
    def __init__(self) -> None:
        self.label: str
        self.description: str
        self.category: str
        self._project: ArcGISProject|None = None
        
    @property
    def project(self) -> ArcGISProject | None:
        """Get the current project that the tool is running in if it exists (otherwise: None)"""
        if self._project is None:
            try:
                self._project = ArcGISProject('CURRENT')
            except Exception:
                pass
        return self._project
    
    def getParameterInfo(self) -> Parameters | list[Parameter]: return Parameters()
    def isLicensed(self) -> bool: return True
    def updateParameters(self, parameters: Parameters | list[Parameter]) -> None: ...
    def updateMessages(self, parameters: Parameters | list[Parameter]) -> None: ...
    def execute(self, parameters: Parameters | list[Parameter], messages: list[Any]) -> None: ...
    def postExecute(self, parameters: Parameters | list[Parameter]) -> None: ...
    
class Parameters(list[Parameter]):
    """Wrap a list of parameters and override the index to allow indexing by name"""
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> Parameter: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Parameter]: ...
    @overload
    def __getitem__(self, key: str, /) -> Parameter: ...
    def __getitem__(self, key: SupportsIndex|slice|str, /) -> Parameter | list[Parameter]:
        if isinstance(key, str):
            _matches = [p for p in self if p.name == key]
            if not _matches:
                raise KeyError(key)
            if len(_matches) == 1:
                return _matches.pop()
            raise KeyError(f'{key} is used for multiple parameters')
        return self[key]