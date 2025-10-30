from __future__ import annotations
from functools import wraps
from pathlib import Path
from types import MappingProxyType

from arcpy import Parameter
from arcpie.featureclass import FeatureClass
from arcpie.project import Layer, Project, Table
from abc import ABC
from collections.abc import Callable
from typing import Any, overload, SupportsIndex

class ToolboxABC(ABC):
    tools: list[type[ToolABC]] = []
    def __init__(self) -> None:
        self.label: str = self.__class__.__name__
        self.alias: str = self.__class__.__name__

class ToolABC(ABC):
    _current_project: Project | None = None
    
    def __init__(self) -> None:
        self.label: str = self.__class__.__name__
        self.description: str = self.__doc__ or 'No Descrption Provided'
        self.category: str | None = None
    
    @property
    def project(self) -> Project:
        """Get the current project that the tool is running in if it exists (otherwise: None)"""
        if ToolABC._current_project is None:
            try:
                ToolABC._current_project = Project('CURRENT')
            except Exception:
                pass
        return ToolABC._current_project # pyright: ignore[reportReturnType]
    
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

# toolify wrapper

# Mapping of Python types to arcpy.Parameter types
_PARAMETER_TYPE_MAP: dict[type, str|list[str]] = {
    int: 'GPLong',
    str: 'GPString',
    float: 'GPDouble',
    FeatureClass: ['DEFeatureClass', 'GPFeatureLayer'],
    Layer: 'GPFeatureLayer',
    Table: 'DETable',
    Project: 'DEWorkspace',
    Path: 'DEFile',
} # Add more as needed

import inspect
from functools import partial

# NOTE: inspect.Parameter is much different from arcpy.Parameter, module level import used to keep them clear

def _build_params(params: MappingProxyType[str, inspect.Parameter]) -> Parameters:
    _parameters: Parameters = Parameters()
    for name, param in params.items():
        p = Parameter(
            name=name,
            displayName=name.replace('_', ' ').title(),
            direction='Input',
            parameterType='Required',
            datatype=_PARAMETER_TYPE_MAP.get(param.annotation, 'GPType'),
        )
        if param.default:
            p.value = param.default
        _parameters.append(p)
    return _parameters

def _read_params(arcpy_params: list[Parameter], func_params: MappingProxyType[str, inspect.Parameter]) -> tuple[tuple[Any], dict[str, Any]]:
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for param in arcpy_params:
        fp = func_params.get(param.name)
        if not fp:
            raise AttributeError(f'toolify: {param.name} not found in wrapped function')
        if not fp.annotation:
            raise AttributeError(f'toolify: Tools created with the toolify wrapper need valid type params!')
        if param.datatype == 'GPFeatureLayer' and fp.annotation == FeatureClass:
            v = FeatureClass[Any].from_layer(param.value)
        else:
            v = fp.annotation(param.value)
        if fp.kind == fp.VAR_KEYWORD:
            kwargs[fp.name] = v
        else:
            args.append(v)
    return tuple(args), kwargs
        

def toolify(toolbox: type[ToolboxABC]):
    """Convert a typed function into a tool for the specified Toolbox class
    
    Args:
        toolbox (type[ToolboxABC]): The toolbox to register the function to
        name (str): The name of the tool
        description (str): An optional description for the tool
    
    Usage:
        ```python
        >>> @toolify(Toolbox)
        >>> def get_counts(feature_class: FeatureClass)
        ```
    """
    def _builder(func: Callable[..., Any]):
        
        @wraps(func)
        def _execute(*args: Any, **kwargs: Any):
            return func(*args, **kwargs)
        
        # Build the class and register it if it doesn't exist
        _label = func.__name__.replace('_', ' ').title()
        _description = func.__doc__
        _class_name = _label.replace(' ', '')
        
        # Don't register the tool again
        if _class_name in map(lambda t: t.__name__, toolbox.tools):
            return _execute
        
        # Build the tool class
        _sig = inspect.signature(func)
        _params = _sig.parameters
        _tool_class: type[ToolABC] = __build_class__(
            lambda: {'label': _label, 'description': _description}, 
            _class_name, ToolABC
        )
        
        # Override getParameterInfo using type annotations
        # Override execute with a wrapper that converts arcpy params to python function call
        setattr(_tool_class, 'getParameterInfo', partial(_build_params, _params))
        
        # Handle parameter converison
        def _passthrough_execution(parameters: Parameters, messages: Any) -> None:
            args, kwargs = _read_params(parameters, _params)
            _execute(*args, **kwargs)
        setattr(_tool_class, 'execute', _passthrough_execution)
        
        toolbox.tools.append(_tool_class)
        
        return _execute
    return _builder
