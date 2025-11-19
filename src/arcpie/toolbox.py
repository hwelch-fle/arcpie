from __future__ import annotations
from functools import wraps
import traceback
from types import MappingProxyType
import inspect
from logging import Logger
from datetime import datetime
import time

from arcpy import Parameter as _Parameter
from arcpie.project import Project
from arcpie._types import ParameterDatatype
from abc import ABC
from collections.abc import Callable
from typing import Any, Literal, TypeVar, overload, SupportsIndex

class Parameter(_Parameter):
    def __init__(self,
                 name: str | None = None, 
                 displayName: str | None = None, 
                 direction: None | Literal['Input', 'Output'] = None, 
                 datatype: str | None | ParameterDatatype = None, 
                 parameterType: None | Literal['Required', 'Optional', 'Derived'] = None, 
                 enabled: bool | None = None, 
                 category: str | None = None, 
                 symbology: str | None = None, 
                 multiValue: bool | None = None) -> None:
        super().__init__(name, displayName, direction, datatype, parameterType, enabled, category, symbology, multiValue)

class ToolboxABC(ABC):
    
    def __init__(self) -> None:
        self.label: str
        self.alias: str
        self.tools: list[type[ToolABC]]

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
    def execute(self, parameters: Parameters | list[Parameter], messages: Any) -> None: ...
    def postExecute(self, parameters: Parameters | list[Parameter]) -> None: ...

_Default = TypeVar('_Default')
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
    
    @overload
    def get(self, key: SupportsIndex, default: _Default=None, /) -> Parameter | _Default: ...
    @overload
    def get(self, key: slice, default: _Default=None, /) -> list[Parameter] | _Default: ...
    @overload
    def get(self, key: str, default: _Default=None, /) -> Parameter | _Default: ...
    def get(self, key: SupportsIndex|slice|str, default: _Default=None, /) -> Parameter | list[Parameter] | _Default:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        match key:
            case str():
                return any(p.name == key for p in self)
            case Parameter():
                return any(p == key for p in self)
            case _:
                return False

# toolify wrapper
# NOTE: inspect.Parameter is much different from arcpy.Parameter, module level import used to keep them clear
def _build_params(params: MappingProxyType[str, inspect.Parameter], types: ParameterTypeMap) -> Parameters:
    _parameters = Parameters()
    for name, param in params.items():
        
        # Build a simple parameter if only type is given, otherwise use the provided Parameter object
        param_type, _ = types.get(name, ('GPType', None))
        if isinstance(param_type, Parameter):
            # Ensure that the parameter name matches the arg name
            param_type.name = name
            p = param_type
        
        else:
            p = Parameter(
                name=name,
                displayName=name.replace('_', ' ').title(),
                direction='Input',
                parameterType='Required',
                datatype=param_type, # pyright: ignore[reportArgumentType]
            )
            p.value = param.default
        _parameters.append(p)
    
    # Match parameter order to types order
    return Parameters([_parameters[name] for name in types if name in _parameters] + [p for p in _parameters if p.name not in types])

def _read_params(arcpy_params: list[Parameter], func_params: MappingProxyType[str, inspect.Parameter], types: ParameterTypeMap) -> tuple[tuple[Any], dict[str, Any]]:
    args: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}
    for arcpy_param in arcpy_params:
        fp = func_params.get(arcpy_param.name)
        if not fp:
            raise AttributeError(f'toolify: {arcpy_param.name} not found in wrapped function')
        _, constructor = types.get(arcpy_param.name, ('GPType', lambda p: p.valueAsText))
        converted_val = constructor(arcpy_param)
        args[fp.name] = converted_val
    
    # Ensure that the function call is passed arguments in the correct order (assuming the type parameters have been moved)
    return tuple(args[name] for name in func_params if name in args), kwargs

from arcpie.utils import print
ParameterTypeMap = dict[str, tuple[ParameterDatatype | list[ParameterDatatype] | Parameter, Callable[[Parameter], Any]]]
def toolify(*tool_registries: list[type[ToolABC]], 
            name: str|None=None, 
            params: ParameterTypeMap|None=None, 
            debug: bool=False, 
            logger: Logger|None=None,
):
    """Convert a typed function into a tool for the specified Toolbox class
    
    Args:
        *tool_registries (list[type[ToolABC]]): The tool registry lists to add this tool to
        name (str): The name of the tool
        params (ParameterTypeMap): A mapping of parameter names to Parameter types and a callable constructor 
            that converts the parameter to the expected value for the function parameter. You can also pass
            a fully formed arcpy.Parameter object as the first item in the tuple instead of a simple type
        debug (bool): Print the converted arguments to the ArcGIS Pro message console (default: False)
        logger (Logger|None): An optional logger to use for logging all runs of the toolified tool
    
    Usage:
        ```python
        >>> @toolify(
        >>>     TOOL_REGISTRY, 
        >>>     name='PDF Exporter', 
        >>>     params={
        >>>         'project': ('DEFile', lambda p: Project(p.valueAsText)),
        >>>         'outfile': ('DEFile', lambda p: Path(p.valueAsText))
        >>>     }
        >>> )
        >>> def export_pdf(project: Project|str='CURRENT', outfile: Path|str='out.pdf') -> None:
        >>>     ...
        ```
    """
    def _builder(func: Callable[..., Any]):
        
        @wraps(func)
        def _execute(*args: Any, **kwargs: Any):
            return func(*args, **kwargs)
        
        # Build the tool class
        _label = func.__name__.replace('_', ' ').title()
        _description = func.__doc__ or 'No Description Provided'
        _class_name = _label.replace(' ', '')
        sig = inspect.signature(func)
        sig_params = sig.parameters
        
        # Handle parameter converison
        def _passthrough_execution(self: ToolABC, parameters: Parameters | list[Parameter], messages: Any) -> None:
            if debug:
                print(f'Executing toolified {func.__name__} via {self.label}')
            args, kwargs = _read_params(parameters, sig_params, params or {})
            start = time.time()
            try:
                if debug:
                    print(f"Using *{args}, **{kwargs}")
                res = _execute(*args, **kwargs)
                end = time.time()
                if logger:
                    logger.info(f'[{datetime.isoformat(datetime.now())}] PASS "{self.label}" ({end-start:0.2f} seconds) [{res}]')
            except Exception as e:
                print(f'Something went wrong!:\n\t{traceback.format_exc()}', severity='ERROR')
                end = time.time()
                if logger:
                    logger.info(f'[{datetime.isoformat(datetime.now())}] FAIL "{self.label}" ({end-start:0.2f} seconds) [{e}] ')
                    
        def _local_build_params(self: ToolABC) -> Parameters | list[Parameter]:
            return _build_params(sig_params, params or {})
        
        def _local_init(self: ToolABC) -> None:
            self.label = name or _label
            self.description = _description

        _tool_class = type(
            _class_name, 
            (ToolABC, ),
            {
                '__init__': _local_init, 
                'getParameterInfo':_local_build_params, 
                'execute': _passthrough_execution
            }
        )

        for registry in tool_registries:
            if _tool_class.__name__ not in map(lambda c: c.__name__, registry):
                registry.append(_tool_class)
                globals()[_class_name] = _tool_class
        return _execute
    return _builder
