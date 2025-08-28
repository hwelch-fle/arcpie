from __future__ import annotations

from arcpy import Parameter
from abc import ABC
from typing import Any, overload, SupportsIndex

class ToolABC(ABC):
    def __init__(self) -> None: ...
    def getParameterInfo(self) -> list[Parameter]: return []
    def isLicensed(self) -> bool: return True
    def updateParameters(self, parameters: list[Parameter]) -> None: ...
    def updateMessages(self, parameters: list[Parameter]) -> None: ...
    def execute(self, parameters: list[Parameter], messages: list[Any]) -> None: ...
    def postExecute(self, parameters: list[Parameter]) -> None: ...
    
class Parameters(list[Parameter]):
    """Wrap a list of parameters and override the index to allow indexing by name"""
    @overload
    def __getitem__(self, key: slice, /) -> list[Parameter]: ...
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> Parameter: ...
    @overload
    def __getitem__(self, key: str, /) -> Parameter: ...
    def __getitem__(self, key): # type: ignore
        if isinstance(key, str):
            _matches = [p for p in self if p.name == key]
            if not _matches:
                raise KeyError(key)
            if len(_matches) == 1:
                return _matches.pop()
            raise KeyError(f'{key} is used for multiple parameters')
        return super().__getitem__(key) # type: ignore
    