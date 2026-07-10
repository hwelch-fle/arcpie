from __future__ import annotations

import datetime as dt
import inspect
import time
import traceback
from collections.abc import Callable
from cProfile import Profile
from datetime import datetime
from functools import wraps
from importlib import import_module, reload
from io import StringIO
from logging import Logger
from pstats import Stats
from traceback import format_exc
from types import MappingProxyType, ModuleType
from typing import Any

from arcpie.parameters import Parameter, Parameters
from arcpie.parameters.custom import *  # noqa: F403
from arcpie.project import Map, Project
from arcpie.types import ParameterDatatype
from arcpie.utils import print


class ToolboxABC:
    def __init__(self) -> None:
        self.label: str
        self.alias: str
        self.tools: list[type[ToolABC]]


class ToolABC:
    def __init__(self) -> None:
        self.label: str = self.__class__.__name__
        self.description: str = self.__doc__ or 'No Descrption Provided'
        self.category: str | None = None

    def getParameterInfo(self) -> Parameters | list[Parameter]: return []
    def isLicensed(self) -> bool: return True
    def updateParameters(self, parameters: Parameters | list[Parameter]) -> None: ...
    def updateMessages(self, parameters: Parameters | list[Parameter]) -> None: ...
    def execute(self, parameters: Parameters | list[Parameter], messages: Any) -> None: ...
    def postExecute(self, parameters: Parameters | list[Parameter]) -> None: ...


class Tool(ToolABC):

    @property
    def project(self) -> Project:
        """Get the current project that the tool is running in if it exists (otherwise: None)"""
        return Project('CURRENT')

    @property
    def active_map(self) -> Map | None:
        if self.project.aprx.activeMap:
            return Map(self.project.aprx.activeMap, parent=self.project)


def profile(*selectors: str):
    def wrapper(func: Callable[..., None]) -> Callable[..., None]:
        @wraps(func)
        def inner(*args: Any, **kwargs: Any):
            err = None
            with Profile() as pr:
                stream = StringIO()
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    err = e
                pr.create_stats()
                stats = Stats(pr, stream=stream)
                stats.sort_stats('time')
                stats.print_stats(*selectors)
                stream.flush()
                stream.seek(0)
                print(stream.read())
            if err:
                raise err
        return inner
    return wrapper


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
              scope: dict[str, Any] | None = None,
              reload_module: bool = False) -> list[type[ToolABC]]:
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
        ...     # Import explicit ToolClass from toolfile (MyTools.py)
        ...     # NOT RECOMMENDED
        ...     'tools.MyTools': ['ToolClassA', 'ToolClassB'],
        ... }
        >>> tools = safe_import(globals(), Tools)
        ```
    """
    tb_tools = [
        _get_tool(module, tool_name, reload_module)
        for module in tools
        for tool_name in tools[module]
    ]
    if scope:
        scope.update({tool.__name__: tool for tool in tb_tools})
    return tb_tools


# toolify wrapper
# NOTE: inspect.Parameter is much different from arcpy.Parameter, module level import used to keep them clear
def _build_params(params: MappingProxyType[str, inspect.Parameter], types: ParameterTypeMap) -> Parameters:
    parameters = Parameters()
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
                datatype=param_type,  # pyright: ignore[reportArgumentType]
            )
            p.value = param.default
        parameters.append(p)

    # Match parameter order to types order
    return Parameters([parameters[name] for name in types if name in parameters] + [p for p in parameters if p.name not in types])


def _read_params(arcpy_params: Parameters | list[Parameter], func_params: MappingProxyType[str, inspect.Parameter], types: ParameterTypeMap) -> tuple[tuple[Any], dict[str, Any]]:
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


ParameterTypeMap = dict[str, tuple[ParameterDatatype | list[ParameterDatatype] | Parameter, Callable[[Parameter], Any]]]


def toolify(*tool_registries: list[type[ToolABC]],
            name: str | None = None,
            params: ParameterTypeMap | None = None,
            debug: bool = False,
            logger: Logger | None = None,
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
        label = func.__name__.replace('_', ' ').title()
        description = func.__doc__ or 'No Description Provided'
        class_name = label.replace(' ', '')
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
                    logger.info(f'[{datetime.isoformat(datetime.now(tz=dt.UTC))}] PASS "{self.label}" ({end - start:0.2f} seconds) [{res}]')
            except Exception as e:
                print(f'Something went wrong!:\n\t{traceback.format_exc()}', severity='ERROR')
                end = time.time()
                if logger:
                    logger.info(f'[{datetime.isoformat(datetime.now(tz=dt.UTC))}] FAIL "{self.label}" ({end - start:0.2f} seconds) [{e}] ')

        def _local_build_params(self: ToolABC) -> Parameters | list[Parameter]:
            return _build_params(sig_params, params or {})

        def _local_init(self: ToolABC) -> None:
            self.label = name or label
            self.description = description

        tool_class = type(
            class_name,
            (ToolABC, ),
            {
                '__init__': _local_init,
                'getParameterInfo': _local_build_params,
                'execute': _passthrough_execution
            }
        )

        for registry in tool_registries:
            if tool_class.__name__ not in (c.__name__ for c in registry):
                registry.append(tool_class)
                globals()[class_name] = tool_class
        return _execute
    return _builder
