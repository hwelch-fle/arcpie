from __future__ import annotations

from importlib import reload, import_module
from traceback import format_exc
from functools import wraps
import traceback
from types import MappingProxyType, ModuleType
import inspect
from logging import Logger
from datetime import datetime
import time

from arcpy import Parameter as _Parameter
from arcpie.project import Map, Project
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
    def __init__(self) -> None:
        self.label: str = self.__class__.__name__
        self.description: str = self.__doc__ or 'No Descrption Provided'
        self.category: str | None = None
    
    def getParameterInfo(self) -> Parameters | list[Parameter]: return Parameters()
    def isLicensed(self) -> bool: return True
    def updateParameters(self, parameters: Parameters | list[Parameter]) -> None: ...
    def updateMessages(self, parameters: Parameters | list[Parameter]) -> None: ...
    def execute(self, parameters: Parameters | list[Parameter], messages: Any) -> None: ...
    def postExecute(self, parameters: Parameters | list[Parameter]) -> None: ...

class Tool(ToolABC):
    _current_project: Project | None = None
    
    @property
    def project(self) -> Project | None:
        """Get the current project that the tool is running in if it exists (otherwise: None)"""
        if __class__._current_project is None:
            try:
                __class__._current_project = Project('CURRENT')
            except Exception:
                pass
        return __class__._current_project
    
    @property
    def active_map(self) -> Map | None:
        if self.project and self.project.aprx.activeMap:
            return Map(self.project.aprx.activeMap, parent=self.project)

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

# Parameter Primitives

class Done(Parameter):
    """An Empty Output parameter that can be used to signal that a tool has completed in Model Builder"""
    def __init__(self) -> None:
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName='Done',
            name='done',
            direction='Output',
            parameterType='Derived',
        )

class Toggle(Parameter):
    """Simple toggle button with a name and default state"""

    def __init__(self, displayName: str, 
                 default: bool=False, 
                 name: str|None=None,
                 category: str|None=None) -> None:
        
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName=displayName,
            # Snake Case the name
            name=name or displayName.lower().replace(' ', '_'),
            parameterType='Required',
            datatype='GPBoolean',
            direction='Input',
            category=category,
        )
        self.value = default

class String(Parameter):
    """Simple string input parameter with filter options and default passthrough"""
    
    def __init__(self, displayName: str, 
                 options: list[str]|None=None, 
                 default: str|None=None,
                 required: bool=True,
                 name: str|None=None,
                 category: str|None=None) -> None:
        
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName=displayName,
            # Snake Case the name
            name=name or displayName.lower().replace(' ', '_'),
            parameterType='Required' if required else 'Optional',
            datatype='GPString',
            direction='Input',
            category=category,
        )
        if self.filter and options:
            self.filter.list = options
        if default is not None:
            self.value = default
            
class StringList(Parameter):
    """Simple string list with default and filter passthroughs"""

    def __init__(self, displayName: str, 
                 options: list[str]|None=None, 
                 defaults: list[str]|None=None,
                 required: bool=True,
                 name: str|None=None,
                 category: str|None=None) -> None:
        
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName=displayName,
            # Snake Case the name
            name=name or displayName.lower().replace(' ', '_'),
            parameterType='Required' if required else 'Optional',
            datatype='GPString',
            direction='Input',
            category=category,
            multiValue=True,
        )
        if self.filter and options:
            self.filter.list = options
        if defaults:
            self.values = defaults

class FilePath(Parameter):
    """Simple filepath input with default and filter passthroughs"""
    def __init__(self, displayName: str, 
                 options: list[str]|None=None, 
                 default: str|None=None,
                 required: bool=True,
                 name: str|None=None,
                 category: str|None=None) -> None:
        
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName=displayName,
            # Snake Case the name
            name=name or displayName.lower().replace(' ', '_'),
            parameterType='Required' if required else 'Optional',
            datatype='DEFile',
            direction='Input',
            category=category,
        )
        if self.filter and options:
            self.filter.list = options
        if default is not None:
            self.value = default

class Integer(Parameter):
    """Simple Integer number parameter with default and filter passthroughs"""
    __name__ = 'Parameter'
    def __init__(self, displayName: str, 
                 options: list[int]|range|None=None, 
                 default: int|None=None,
                 required: bool=True,
                 name: str|None=None,
                 category: str|None=None) -> None:
        
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName=displayName,
            # Snake Case the name
            name=name or displayName.lower().replace(' ', '_'),
            parameterType='Required' if required else 'Optional',
            datatype='GPLong',
            direction='Input',
            category=category,
        )
        if self.filter and options:
            if isinstance(options, range):
                if options.step:
                    self.filter.list = list(options)
                else:
                    self.filter.type = 'Range'
                    self.filter.list = [options.start, options.stop]
            else:
                self.filter.list = options
        if default is not None:
            self.value = default

class Double(Parameter):
    """Simple Double/Float parameter with default and filter passthroughs"""
    __name__ = 'Parameter'
    def __init__(self, displayName: str, 
                 options: list[float]|None=None, 
                 default: float|None=None,
                 required: bool=True,
                 name: str|None=None,
                 category: str|None=None) -> None:
        
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName=displayName,
            # Snake Case the name
            name=name or displayName.lower().replace(' ', '_'),
            parameterType='Required' if required else 'Optional',
            datatype='GPDouble',
            direction='Input',
            category=category,
        )
        if self.filter and options:
            self.filter.list = options
        if default is not None:
            self.value = default

class FeatureLayer(Parameter):
    """Simple Feature Layer parameter with filter and default passthroughs"""
    __name__ = 'Parameter'
    def __init__(self, displayName: str, 
                 options: list[str]|None=None,
                 default: str|None=None,
                 required: bool=True,
                 name: str|None=None,
                 allow_create: bool=False,
                 category: str|None=None) -> None:
        
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName=displayName,
            # Snake Case the name
            name=name or displayName.lower().replace(' ', '_'),
            parameterType='Required' if required else 'Optional',
            datatype='GPFeatureLayer',
            direction='Input',
            category=category,
        )
        if self.filter and options:
            self.filter.list = options
        if default is not None:
            self.value = default
        if allow_create:
            self.controlCLSID = '{60061247-BCA8-473E-A7AF-A2026DDE1C2D}'

class Folder(Parameter):
    """Simple Feature Layer parameter with filter and default passthroughs"""
    __name__ = 'Parameter'
    def __init__(self, displayName: str, 
                 options: list[str]|None=None,
                 default: str|None=None,
                 required: bool=True,
                 name: str|None=None,
                 category: str|None=None) -> None:
        
        self.__class__.__name__ =  __name__ = 'Parameter'
        super().__init__(
            displayName=displayName,
            # Snake Case the name
            name=name or displayName.lower().replace(' ', '_'),
            parameterType='Required' if required else 'Optional',
            datatype='DEFolder',
            direction='Input',
            category=category,
        )
        if self.filter and options:
            self.filter.list = options
        if default is not None:
            self.value = default

def _placeholder_tool(tool_name: str, exception: Exception, traceback: str) -> type[ToolABC]:
    """ Higher order function for creating a tool class that represents a broken tool. """
    class _BrokenImport(ToolABC):
        __name__ = f"{tool_name}_BrokenImport"
        __exc__ = exception
        def __init__(self):
            self.category = "Broken Tools (Read Description for More Info)"
            self.label = f"{tool_name} - {exception}"
            self.alias = self.label.replace(" ", "")
            self.description = traceback
    return _BrokenImport

def _import_mod(module_name: str, tool_name: str) -> ModuleType:
    try:
        # Tool File with matching class name
        mod = import_module(f'{module_name}.{tool_name}')
    except ImportError:
        # Direct class import
        mod = import_module(module_name)
    return mod

def _get_tool(module_name: str, tool_name: str, reload_module: bool) -> type[ToolABC]:
    # Last component is always the ToolClass name
    try:
        mod = _import_mod(module_name, tool_name)
        if reload_module:
            reload(mod)
        return getattr(mod, tool_name)
    except Exception as e:
        return _placeholder_tool(tool_name, e, format_exc(limit=1))

def safe_load(tools: dict[str, list[str]],
              *,
              scope: dict[str, Any]|None=None,
              reload_module: bool=False) -> list[type[ToolABC]]:
    """Safely load in tools to a toolbox placing all failed imports in a `Broken Tools` category
    
    Args:
        tools (dict[str, list[str]]): A mapping of tool modules to tool files or tool classes in a module
        scope (dict[str, Any]): The `globals()` dict for the Toolbox scope (required so loading happend in Toolbox scope)
        reload_module (bool): Reload the module after importing (default: False)
    
    Returns:
        ( list[type[ToolABC]] ): A list of tool classes
    
    Note:
        To isolate bugs in a large toolbox, it is reccomended that you use file/module level importing
        with matching toolclass names (e.g. `Tool.py -> class Tool`) where the Tool mapping is:
            `Tools = {'tools': ['Tool.py']}`
        instead of:
            `Tools = {'tools.Tool': ['Tool']}`
        This allows the import to fail softly on a single tool instead of breaking imports for all other
        toolclasses in that `Tool.py` module.
        
        Any tools with errors will be placed in a `Broken Tools` category in the toolbox with the full
        stack trace placed in its description field. The final component of the exception will be placed
        at the end of the tool label for easy debugging.
    
    Example:
        ```python
        >>> # File import
        >>> Tools = {
        ...     # import matching class from named file in a submodule (tools)
        ...     # where ToolFileA is ToolFileA.py with a class ToolFileA
        ...     'tools': ['ToolFileA', 'ToolFileB'],
        ...     
        ...     # Import explicit ToolClass from toolfile (MyTools.py)
        ...     # NOT RECOMMENDED
        ...     'tools.MyTools': ['ToolClassA', 'ToolClassB'],
        ... }
        >>> tools = safe_import(globals(), Tools)
        ```
    """
    _tools = [
        _get_tool(module, tool_name, reload_module)
        for module in tools
        for tool_name in tools[module]
    ]
    if scope:
        scope.update({tool.__name__: tool for tool in _tools})
    return _tools
    

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
