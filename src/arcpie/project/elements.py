"""Python style interface for arcpy.mp submodule

This module attempts to wrap the mp classes and functions in a way that provides a more OO
approach to navigating and managing ArcGISProject objects and their associated elements.

All list* methods are replaced with ElementList properties that allow indexing on
longName/name/URI/regex name match.

All elements when accessed from a Project root will have a parent attribute that allows
traversal of the CIM/DOM

Objects that returned <obj> | None have been mostly replaced with Exceptions.
Anywhere that this has been done, a boolean property has been included to
allow checking the property before accessing it. If you expect the property
to exist, allow the Exception to be raised and handle that properly in your implementation.

Overhead has been kept as low as possible. There is agressive cache usage, so manual invalidation
using Element.refresh is required if you modify an element outside this framework.

Some elements have no CIM properties in arcpy, but the aprx file actually contains their raw CIM definition.
These elements allow you to access the CIM in read-only mode and will not allow you to set it.

Example:
    ```python
    >>> with Project('my-project.aprx') as prj:
    ...     for map in prj.maps['Plan*']:
    ...         for lay in map.layers['*Route']:
    ...             # when using `with Project() as ...`
    ...             # Project.save is called on __exit__
    ...             if lay.name == 'Proposed Route':
    ...                 lay.visible = False
    ...             print(f'{lay.name} (2024+): {len(lay)}')
    ...             with lay.query_as('YEAR >= 2024'):
    ...                 print(f'{lay.name} (2024+): {len(lay)}')
    Proposed Route: 216
    Proposed Route (2024+): 65
    Final Route: 673
    Final Route (2024+): 175
    ```
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from copy import copy
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    Never,
    Protocol,
    Self,
    SupportsIndex,
    TypedDict,
    TypeIs,
    Unpack,
    cast,
    overload,
    runtime_checkable,
)
from zipfile import ZIP_DEFLATED, ZipFile

import arcpy._mp as mpt
import arcpy.charts as cht
import arcpy.cim as cim
import arcpy.mp as mp
import httpx
from arcpy import Extent, Point, Polygon, Polyline, SpatialReference
from arcpy.cim.cimloader import jsontocim
from arcpy.cim.cimloader.cimtojson import CimJsonEncoder as CIMJsonEncoder

# Remove after testing
try:
    from ..database import Dataset
except ImportError:
    import sys
    root = str(Path(__file__).parent.parent.parent.resolve())
    sys.path.append(root)

import arcpie.project.formats as fmts
from arcpie import (
    database as db,
    featureclass as fc,
)

if TYPE_CHECKING:
    # Remove after testing
    from arcpie import (
        database as db,
        featureclass as fc,
    )
else:
    mpt = cim = type(
        '_mock', (object, ),
        {'__getattr__': lambda s, n: object}
    )()

from arcpy._symbology import Symbology
from arcpy.metadata import Metadata

__all__ = ('Project',)


# QOL str/repr patches for SpatialReference
# Seeing WKID and Name are more useful than memory address
SpatialReference.__str__ = lambda self: self.name
SpatialReference.__repr__ = lambda self: f'{type(self).__name__}({self.factoryCode})'


type MPElement = (
    mpt.ArcGISProject
    | mpt.Map
    | mpt.MapView
    | mpt.Layer
    | mpt.Table
    | mpt.ElevationSurface
    | mpt.Layout
    | mpt.MapFrame
    | mpt.MapSeries
    | mpt.Bookmark
    | mpt.BookmarkMapSeries
    | mpt.Report
    | mpt.ElevationSource
    | mpt.StyleItem
    | mpt.LabelClass
    # LayoutElements
    | mpt.MapSurroundElement
    | mpt.TableFrameElement
    | mpt.GraphicElement
    | mpt.GroupElement
    | mpt.LegendElement
    | mpt.LegendItem
    | mpt.PictureElement
    | mpt.TextElement
    # ReportSections
    | mpt.ReportSection
    | mpt.ReportLayoutSection
    | None
)

type Chart = (
    cht.Bar
    | cht.Box
    | cht.CalendarHeat
    | cht.Combo
    | cht.DataClock
    | cht.Histogram
    | cht.Line
    | cht.MatrixHeat
    | cht.Pie
    | cht.QQPlot
    | cht.Scatter
    | cht.ScatterMatrix
)
ChartType = Literal[
    'Bar',
    'Box',
    'CalendarHeat',
    'Combo',
    'DataClock',
    'Histogram',
    'Line',
    'MatrixHeat',
    'Pie',
    'QQPlot',
    'Scatter',
    'ScatterMatrix',
]
ChartTypes: tuple[ChartType, ...] = ChartType.__args__


class ChartMap(TypedDict):
    Bar: list[cht.Bar]
    Box: list[cht.Box]
    CalendarHeat: list[cht.CalendarHeat]
    Combo: list[cht.Combo]
    DataClock: list[cht.DataClock]
    Histogram: list[cht.Histogram]
    Line: list[cht.Line]
    MatrixHeat: list[cht.MatrixHeat]
    Pie: list[cht.Pie]
    QQPlot: list[cht.QQPlot]
    Scatter: list[cht.Scatter]
    ScatterMatrix: list[cht.ScatterMatrix]


# Type unions that allow arcpie Elements and mp Elements to be used interchangeably
type MapLike = Map | mpt.Map
type MapViewLike = MapView | mpt.MapView
type LayerLike = Layer | mpt.Layer
type GroupLayerLike = GroupLayer | mpt.Layer
type TableLike = Table | mpt.Table
type ElevationSurfaceLike = ElevationSurface | mpt.ElevationSurface
type LayoutLike = Layout | mpt.Layout
type MapFrameLike = MapFrame | mpt.MapFrame
type MapSeriesLike = MapSeries | mpt.MapSeries
type BookmarkLike = Bookmark | mpt.Bookmark
type BookmarkMapSeriesLike = BookmarkMapSeries | mpt.BookmarkMapSeries
type ReportLike = Report | mpt.Report
type ElevationSourceLike = ElevationSource | mpt.ElevationSource
type StyleItemLike = StyleItem | mpt.StyleItem
type LabelClassLike = LabelClass | mpt.LabelClass
type MapSurroundElementLike = MapSurroundElement | mpt.MapSurroundElement
type TableFrameElementLike = TableFrameElement | mpt.TableFrameElement
type GraphicElementLike = GraphicElement | mpt.GraphicElement
type GroupElementLike = GroupElement | mpt.GroupElement
type LegendElementLike = LegendElement | mpt.LegendElement
type LegendItemLike = LegendItem | mpt.LegendItem
type PictureElementLike = PictureElement | mpt.PictureElement
type TextElementLike = TextElement | mpt.TextElement
type ReportSectionLike = ReportSection | mpt.ReportSection
type ReportLayoutSectionLike = ReportLayoutSection | mpt.ReportLayoutSection


def _get_props(elem: MPElement, *attrs: str) -> dict[str, Any]:
    """Lazily get properties from a Mapping element.

    Args:
        elem: The element to find the properties for
        *attrs: An optional filter to apply (some props are slow to access)

    Note:
        Some elements have properties that raise when accessed, this function will
        ignore those properties as if they don't exist. It is also best to use this
        with an attr filter since many props can have dozen+ ms access time
        (e.g. `defaultCamera` and `defaultView`)
    """
    props = dict[str, Any]()
    broken = not str(getattr(elem, 'URI', '')).startswith('CIMPATH=')
    if attrs:
        attrs = tuple(set(attrs).intersection(dir(elem)))
    else:
        attrs = tuple(dir(elem))
    for attr in attrs:
        # Seems that accessing the `time` attribute of malformed layers (no CIMPATH)
        # will cause some sort of segfault in ArcPro and crash the process with no warning
        if broken and attr == 'time':
            continue
        with suppress(AttributeError, RuntimeError, SystemError):
            props[attr] = getattr(elem, attr)
    return props


def _get_name(elem: MPElement) -> str:
    """Try all possible ways to get a unique name from a mapping element.
     Fallback to ``ID:{hex(id(elem))}`` if `longName` and `name` do not exist.
    """
    props = _get_props(elem, 'longName', 'name', 'URI')
    return str(
        props.get('longName')
        or props.get('name')
        or props.get('URI')
        or f'ID:{hex(id(elem))}'
    )


def _get_uri(elem: MPElement) -> str:
    """Try all possible ways to get a URI from a mapping element.
     Fallback of ``NO_URI:{type(elem)}<{get_name(elem)}@{hex(id(elem))}>`` is used if uRI cannot be determined.
    """
    props = _get_props(elem, 'URI')
    fallback_uri = f'NO_URI:{type(elem).__name__}<{_get_name(elem)}@{hex(id(elem))}>'
    if (uri := props.get('URI')):
        return str(uri).replace('CIMPATH=', '')
    if not (arc_object := getattr(elem, '_arc_object', None)):
        return fallback_uri
    if not (cim_str := getattr(arc_object, 'GetCimJSONString', None)):
        return fallback_uri
    cim_dict: dict[str, Any] | None = json.loads(cim_str() or '{}')
    if not isinstance(cim_dict, dict):
        return fallback_uri
    # PictureElements actually have a hidden uri if the image is bundled
    # Just try and get it if possible
    if 'uRI' in cim_dict:
        return str(cim_dict['uRI']).replace('CIMPATH=', '')
    if 'graphic' in cim_dict:
        with suppress(Exception):
            return str(cim_dict['graphic']['referenceURI']).replace('CIMPATH=', '')
    return fallback_uri


# Used when we need to consume a callable but none exists
def _noop(*args: Any, **kwargs: Any) -> None: ...

# Allow any class that has these attributes to be used in geometry operations
@runtime_checkable
class HasCentroid(Protocol):
    centroid: Point
    trueCentroid: Point
    spatialReference: SpatialReference


@runtime_checkable
class HasExtent(Protocol):
    extent: Extent
    spatialReference: SpatialReference


# mapping elements with no get/set CIM methods
# we use the typedef for hinting and the name list for runtime
# validation since mp does not actually export these at runtime
type _NoCIM = (
    mpt.ArcGISProject
    | mpt.MapView
    | mpt.ElevationSurface
    | mpt.Bookmark
    | mpt.ElevationSource
    | mpt.StyleItem
    | mpt.ReportSection
    | mpt.ReportLayoutSection
    | mpt.LegendItem
    | None
)


def _cimless(obj: Any) -> TypeIs[_NoCIM]:
    return type(obj).__name__ in frozenset({
            'ArcGISProject',
            'MapView',
            'ElevationSurface',
            'Bookmark',
            'ElevationSource',
            'StyleItem',
            'ReportSection',
            'ReportLayoutSection',
        })


# Element takes Base arcpy.mp element, cim definition (from `getDefinition`) and parent type
class Element[MPElem: MPElement, CIMDef, Parent: Element | None = None]:
    def __init__(self, elem: MPElem | Element[MPElem, CIMDef, Parent], parent: Parent | None = None) -> None:
        if isinstance(elem, type(self)):
            elem = elem.elem
        self.elem = cast(MPElem, elem)
        self._elemattrs = set(dir(elem))
        self.parent = parent
        self.children = ElementList[Element[Any, Any, Any]]()
        self.mp_type = type(elem)
        self.mp_type_name = self.mp_type.__name__
        self.uri = _get_uri(self.elem)
        self.unique_name = f'{self.name}:{self.uri}'
        self.cache = dict[str, Any]()
        self._cache_enabled = True
        if parent is not None:
            parent.children.append(self)
            self._cache_enabled = parent.cache_enabled

    @property
    def name(self):
        """The name of the element (either `longName`, `name`, `URI`, or `ID:{hex(id(elem)}`)"""
        return _get_name(self.elem)

    @name.setter
    def name(self, name: str) -> None:
        if hasattr(self.elem, 'name'):
            self.elem.name = name  # type: ignore

    def _cached[T](self, attr: str, default: Callable[[], T] | None = None) -> T:
        """Return a shallow copy of the value stored in the cache (or set the cache)."""
        if default and not self._cache_enabled:
            return default()
        if item := self.cache.get(attr):
            return copy(item)
        return copy(self.cache.setdefault(attr, default() if default else None))

    def refresh(self, *props: str) -> None:
        """Refresh all cached elements.

        By default, all list* function results are cached and copies of those
        objects are returned. If a state change occurs (elements are added/removed)
        refresh() should be called.

        Args:
            props: varargs for all cache entries to clear (default: `<all>`)
        """
        if not props:
            self.cache.clear()
            self.children.clear()
            return
        orphans = set[Element[Any, Any, Self]]()
        for prop in props:
            for orphan in self.cache.pop(prop, []):
                orphans.add(orphan)
        self.children = ElementList(
            chld for chld in self.children
            if chld not in orphans
        )

    @property
    def cache_enabled(self) -> bool:
        """See if cache is enabled for this Element

        Element cache sits on top of the base ArcGIS cache and will not be invalidated
        unless this flag is set to `False` or an explicit `refresh` call is made

        When accessing element items in a tight loop, this cache can be
        ~100x faster (~1us vs ~100us) at the cost of de-sync if lots of changes are
        being made to the element children/state (e.g. adding maps/layers in a loop).

        Methods implemented on the arcpie wrappers will manage this cache, but changes
        that happen outside arcpie will require manual cache invaludation.
        """
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, cache_enabled: bool) -> None:
        """Enable or disable cache for this Element (default: enabled)"""
        self._cache_enabled = cache_enabled
        for child in self.children:
            child.cache_enabled = cache_enabled
        if not cache_enabled:
            # Invalidate the cache when it is explicitly disabled
            self.cache.clear()

    @property
    def cim_type(self) -> type[CIMDef]:
        """The CIM object returned by `getDefinition('V3')`"""
        return type(self.cim)

    @property
    def cim_type_name(self) -> str:
        """The name of the CIM object returned by `getDefinition('V3')`"""
        return self.cim_type.__name__

    @property
    def cim(self) -> CIMDef:
        """The CIM object for the Element"""
        return self.get_cim()

    @cim.setter
    def cim(self, cim: CIMDef) -> None:
        self.set_cim(cim)

    def get_cim(self) -> CIMDef:
        """Get the CIM definition for the Element."""
        elem = self.elem
        # Try really hard to access the CIM definition,
        # even if it's not available in the public API
        if _cimless(elem):
            if arc_object := getattr(elem, '_arc_object', None):
                if getter := getattr(arc_object, 'GetCimJSONString', None):
                    with suppress(Exception):
                        assert callable(getter)
                        return jsontocim.GetJSONTypeOBJ(json.loads(cast(str, getter())))  # type: ignore
            raise AttributeError(f'{type(self).__name__} has no implemented CIM getter')
        return cast(CIMDef, elem.getDefinition('V3'))

    def set_cim(self, cim: CIMDef, _force: bool = False) -> None:
        """Set a CIM definition for an Element set `_force` to `True` to allow CIMType change.
        (must be valid instance of original CIM type by default)"""
        elem = self.elem
        target_type = type(self.cim)
        if not isinstance(cim, target_type) and not _force:
            raise ValueError(f'{self} expects cim type {target_type.__name__}, got {type(cim).__name__}')
        if _cimless(elem):
            if arc_object := getattr(elem, '_arc_object', None):
                if setter := getattr(arc_object, 'SetCimJSONString', None):
                    with suppress(Exception):
                        assert callable(setter)
                        setter(cim)
                        return
            raise AttributeError(f'{type(self).__name__} has no implemented CIM setter')
        elem.setDefinition(cast(str, cim))  # Signature here is wrong CIM obj OR string name

    @property
    def cim_dict(self) -> dict[str, Any]:
        """The Element CIM definition as a Python dictionary"""
        return json.loads(json.dumps(self.cim or '{}', cls=CIMJsonEncoder))

    @property
    def short_name(self) -> str:
        """A short name for the Element (`elem.name`) if one exists otherwise the same as `self.name`"""
        return getattr(self.elem, 'name', self.name)

    if not TYPE_CHECKING:  # Allow runtime access to base attrs
        def __getattr__(self, name: str):
            try:
                return super().__getattribute__(name)
            except AttributeError:
                if name in self._elemattrs:
                    return getattr(self.elem, name)
                raise

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.short_name if not self.name.startswith('ID:0x') else self.parent})'

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, type(self)) and self.elem == other.elem) or super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.elem._arc_object)  # type: ignore

    @classmethod
    def diff(cls, a: Self, b: Self, *, outfile: Path | str | None = None) -> str:
        """Generate a diff of two Elements using `cim_dict` and `difflib.unified_diff`

        Args:
            a: The A Element of the diff
            b: The B Element of the diff
            outfile: An optional `.diff` file to write the diff to
        """
        diff = '\n'.join(difflib.unified_diff(
            json.dumps(a.cim_dict, indent=2).split('\n'),
            json.dumps(b.cim_dict, indent=2).split('\n'),
            fromfile=f'{a.name} (a)',
            tofile=f'{b.name} (b)',
        ))
        if outfile:
            Path(outfile).with_suffix('.diff').write_text(diff)
        return diff


# Subclass Key/Value/Index error so you can catch this with normal Exception subclasses
class UnwrapError(KeyError, ValueError, IndexError):
    def __init__(self, expects: int, have: int):
        super().__init__(f'expected {expects}: have {have}')
        self.expects = expects
        self.have = have


class ElementList[E: Element[Any, Any, Any]](list[E]):
    """Simple list wrapper that allows accessing elements from a list by name/uRI."""

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> E: ...
    @overload
    def __getitem__(self, s: slice, /) -> Self: ...
    @overload
    def __getitem__(self, key: str, /) -> Self: ...
    @overload
    def __getitem__(self, key: re.Pattern[str], /) -> Self: ...
    def __getitem__(self, key: SupportsIndex | slice | str | re.Pattern[str]) -> Self | E:
        if isinstance(key, (str, re.Pattern)):
            matches = type(self)()
            # Attempt direct name/uri/short name match
            if isinstance(key, str):
                matches = type(self)(
                    elem for elem in self
                    if key in
                        (
                            # longName ?
                            elem.name,
                            # CIM path
                            elem.uri,
                            # shortName
                            elem.name.split('\\')[-1],
                        )
                )
            # Fallback to checking re.Pattern or a "pattern like" string
            if not matches and (
                isinstance(key, re.Pattern)
                or any(op in key for op in ('*', '.', '^', '?', '$', '|'))
            ):
                pat = re.compile(key)
                matches = type(self)(
                    elem for elem in self
                    if pat.search(elem.name)
                )
            if not matches:
                raise IndexError(f'No elements with name {key} found')
            return matches
        if isinstance(key, slice):
            return type(self)(super().__getitem__(key))
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        try:
            return self.get(key) is not None
        except Exception:
            return False

    def filter(self, cond: Callable[[E], bool]) -> Self:
        """Filter elements in the list using the provided function"""
        return type(self)(e for e in self if cond(e))

    @overload
    def __add__(self, value: Sequence[E], /) -> Self: ...
    @overload
    def __add__[S: Element[Any, Any, Any]](self, value: Sequence[S], /) -> ElementList[S | E]: ...
    def __add__[S: Element[Any, Any, Any]](self, value: Sequence[E] | Sequence[S], /) -> ElementList[S | E] | ElementList[E]:  # type: ignore (we don't want non-Elements)
        return ElementList(super().__add__(list(value)))

    @overload
    def get[D](self, i: SupportsIndex, /, default: D = ...) -> E | D: ...
    @overload
    def get[D](self, s: slice, /, default: D = ...) -> Self | D: ...
    @overload
    def get[D](self, key: str, /, default: D = ...) -> Self | D: ...
    @overload
    def get[D](self, i: re.Pattern[str], /, default: D | None = None) -> Self | D: ...
    def get[D](self, key: SupportsIndex | str | re.Pattern[str] | slice, /, default: D = None) -> Self | E | D:
        """Get the item from the list but return the default if it doesn't exist"""
        try:
            return self[key]
        except IndexError:
            return default

    def copy(self) -> ElementList[E]:
        return type(self)(super().copy())

    @overload
    def unwrap(self, expects: Literal[1] = 1, /, *, panic: Literal[True] = True) -> E: ...  # type: ignore
    @overload
    def unwrap(self, expects: Literal[1] = 1, /, *, panic: Literal[False] = False) -> E | KeyError: ...
    @overload
    def unwrap(self, expects: int = ..., /, *, panic: Literal[True] = True) -> Self: ...
    @overload
    def unwrap(self, expects: int = ..., /, *, panic: Literal[False] = False) -> Self | KeyError: ...
    def unwrap(self, expects: int = 1, /, *, panic: bool = True) -> Self | E | KeyError:
        """Unwrap the ElementList and check to see if it has the expected number of items

        Args:
            expects: The number of elements you expect to have (default: `1`)
                if the number of expected elements is 1, a single element is requrned.
            panic: If set to `False`, the Exception is returned instead of raised (default: `True`)

        Returns:
            Element: if `expects` is 1
            Self: if `expects` is > 1 and `expects` == length
            UnwrapError: (KeyError subclass) if `expects` != length and `panic` is `False`

        Raises:
            UnwrapError: (KeyError subclass) `expects` != length and `panic` is `True`

        Example:
            ```python
            >>> prj = Project(...)
            >>> mp = prj.maps['My Map'].unwrap()
            >>> mp
            Map(My Map)
            >>> mp = prj.maps['Duplicate Map'].unwrap()
            Traceback (most recent call last):
                File ...
            UnwrapError: 'expected 1: have 2'
            >>> mp = prj.maps['Duplicate Map'].unwrap(panic=False)
            >>> mp
            UnwrapError('expected 1: have 2')
            >>> mp = prj.maps['Duplicate Map'].unwrap(expects=2)
            >>> mp
            [Map(Duplicate Map), Map(Duplicate Map)]
            ```
        """
        if len(self) == expects:
            if expects == 1:
                return self[0]
            else:
                return self
        exc = UnwrapError(expects, len(self))
        if not panic:
            return exc
        else:
            raise exc


class ToolboxConf(TypedDict):
    toolboxPath: Path | str
    isDefaultToolbox: bool


class FolderConf(TypedDict):
    alias: str
    connectionString: Path | str
    isHomeFolder: bool


class ReportFieldConf(TypedDict):
    fieldName: str
    sortInfo: Literal['ASC', 'DESC', 'NONE']
    groupField: bool


class ReportStatConf(TypedDict):
    fieldName: str
    statistic: Literal['COUNT', 'MEAN', 'SUM', 'STD_DEV', 'MAX', 'MIN']


class _ClosedProject:
    """Replaces Project.elem when Project.close is called"""
    def __init__(self, path: str) -> None:
        self.isReadOnly = True
        self.filePath = path

    def __getattr__(self, name: str) -> Any:
        raise PermissionError(f'{self.filePath} is closed')


class APRXNode(TypedDict):
    NodeId: int
    NodeType: str
    FileName: str
    ChildNodeIds: str


class ProjectNode(TypedDict):
    type: str
    path: str
    name: str
    children: set[int]


class Project(Element[mpt.ArcGISProject, cim.CIMGISProject]):
    """ArcGISProject wrapper

    [ESRI Documentation](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/arcgisproject-class.html)

    A wrapper for the `mp.ArcGISProject` class that converts all child elements to wrapped elements.
    All methods of base object can be accessed using `project.elem` or directly called if they have
    no implemented wrapper version.

    Example:
        ```python
        >>> prj = Project(...)
        >>> prj.maps
        [Map(Map 1), Map(Map 2)]
        >>> with prj:
        ...     prj.maps['Map 1'].unwrap().set_name('My Map')
        >>> prj.reopen()
        >>> prj.maps
        [Map(My Map), Map(Map 2)]
        ```
    """

    # Need to override Element.__init__ since ArcGISProject objects are special
    def __init__(self, aprx: Path | str | Literal['CURRENT'] | None = None, *, cached: bool = True) -> None:
        self._aprx = aprx or 'CURRENT'
        self._is_open = False
        self.elem = None  # type: ignore
        self.open()
        assert self.elem is not None, f'Unable to open {aprx}'
        self.elem: mp.ArcGISProject
        super().__init__(self.elem, None)
        self.cache_enabled = cached
        self._is_open = True

    @property
    def name(self) -> str:
        """The final path component of the Project file including the .aprx suffix"""
        return self.path.name

    @name.setter
    def name(self, name: Never) -> Never:  # type: ignore
        """Project name cannot be set"""

    # Backwards compat (deprecate eventually)
    @property
    def aprx(self) -> mp.ArcGISProject:
        """Alias for `elem`"""
        return self.elem

    def open(self) -> None:
        """Open the project if it is not open (stores an `ArcGISProject` instance in `elem`)"""
        # Don't re-initialize a CURRENT project
        if self.is_current and self._is_open:
            return
        prj = mp.ArcGISProject(str(self._aprx))
        self._is_open = True
        self.elem = prj

    def close(self) -> None:
        """Close the Project if is is open (cannot close `CURRENT` project)"""
        # Can't close a CURRENT project
        if self.is_current or not self.is_open:
            return
        self.elem = _ClosedProject(str(self.path))  # type: ignore
        self.refresh()
        self.children.clear()
        self._is_open = False

    def reopen(self) -> None:
        """Close and re-open a Project"""
        if self.is_current:
            return
        self.close()
        self.open()

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.path})'

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_current(self) -> bool:
        return self._aprx == 'CURRENT'

    @property
    def is_read_only(self) -> bool:
        return self.elem.isReadOnly

    @property
    def date_saved(self) -> datetime:
        """The datetime the project was last saved"""
        return self.elem.dateSaved

    @property
    def version(self) -> str:
        """A string representation of the ArcGIS Pro version the project was created/saved in
        ``MAJOR.MINOR.PATCH``
        """
        return self.elem.documentVersion

    # ArcGISProject CIM is not directly available and needs to be loaded from
    # the raw GISProject.json file in the aprx zip directory
    @property
    def cim(self) -> cim.CIMGISProject:
        """Load the CIM data from the `GISProject.json` file in the `aprx` zip directory"""
        with ZipFile(self.path) as zf, zf.open('GISProject.json') as cim:
            return jsontocim.GetJSONTypeOBJ(json.load(cim))  # type: ignore

    @cim.setter
    def cim(self, cim: Never) -> Never:  # type: ignore
        """Project CIM is read only"""

    @property
    def path(self) -> Path:
        """A `Path` object pointed at the Project `aprx` file"""
        return Path(self.elem.filePath)

    @property
    def home(self) -> Path:
        """A `Path` object pointed at the Project Home Folder"""
        return Path(self.elem.homeFolder)

    @home.setter
    def home(self, path: Path | str) -> None:
        self.elem.homeFolder = str(path)

    @property
    def maps(self) -> ElementList[Map]:
        """An `ElementList` of Maps in the Project.
        By default this property is cached on first load
        """
        return self._cached('maps',
            lambda: ElementList(
                Map(map, self)
                for map in self.elem.listMaps()
            )
        )

    @property
    def layouts(self) -> ElementList[Layout]:
        """An `ElementList` of Layouts in the Project.
        By default this property is cached on first load
        """
        return self._cached('layouts',
            lambda: ElementList(
                Layout(layout, self)
                for layout in self.elem.listLayouts()
            )
        )

    @property
    def reports(self) -> ElementList[Report]:
        """An `ElementList` of Reports in the Project.
        By default this property is cached on first load
        """
        return self._cached('reports',
            lambda: ElementList(
                Report(report, self)
                for report in self.elem.listReports()
            )
        )

    @property
    def styles(self) -> ElementList[Style]:
        """An `ElementList` of Styles in the Project.
        By default this property is cached on first load
        """
        return self._cached('styles',
            lambda: ElementList(
                Style(st, self)
                for st in self.elem.styles
            )
        )

    @styles.setter
    def styles(self, styles: Iterable[str | Path | Style]) -> None:
        self.elem.updateStyles([str(style) for style in styles])
        self.refresh('styles')

    @property
    def toolboxes(self) -> list[ToolboxConf]:
        """A list of Toolboxes in the Project."""
        return cast(list[ToolboxConf], self.elem.toolboxes)

    @toolboxes.setter
    def toolboxes(self, toolboxes: Iterable[ToolboxConf]):
        self.elem.updateToolboxes(
            [
                {
                    'toolboxPath': str(tb['toolboxPath']),
                    'isDefaultToolbox': tb['isDefaultToolbox'],
                }
                for tb in toolboxes
            ],
            validate=True
        )

    @property
    def folder_connections(self) -> list[FolderConf]:
        """A list of Folder Connections in the Project"""
        return cast(list[FolderConf], self.elem.folderConnections)

    @folder_connections.setter
    def folder_connections(self, folders: Iterable[FolderConf]) -> None:
        self.elem.updateFolderConnections(
            [
                {
                    'alias': fldr['alias'],
                    'connectionString': str(fldr['connectionString']),
                    'isHomeFolder': fldr['isHomeFolder'],
                }
                for fldr in folders
            ],
            validate=True
        )

    @property
    def active_map(self) -> Map:
        """The active map for the Project. Check for active map using `has_active_map` to avoid `AttributeError`
        Raises:
            AttributeError: If the Project has no active map
        """
        active = self.elem.activeMap
        if not active:
            raise AttributeError(f'{self} has no active map')
        return Map(active, self)

    @property
    def has_active_map(self) -> bool:
        """Check to see if the Project has an active map set"""
        return bool(self.elem.activeMap)

    @property
    def active_view(self) -> MapView | Layout | Report:
        """The active map for the View. Check for active map using `has_active_view` to avoid `AttributeError`
        Raises:
            AttributeError: If the Project has no active view
        """
        active = self.elem.activeView
        view_type = type(active).__name__
        if view_type == 'MapView':
            return MapView(cast(mpt.MapView, active), self)
        if view_type == 'Layout':
            return Layout(cast(mpt.Layout, active), self)
        if view_type == 'Report':
            return Report(cast(mpt.Report, active), self)

        raise AttributeError(f'{self} has no avtive view')

    @active_view.setter
    def active_view(self, view: MapViewLike | LayoutLike | ReportLike) -> None:
        if not self.is_current:
            raise AttributeError('Can only set the view on an actve project initialized with "CURRENT"')
        if isinstance(view, Element):
            view = view.elem
        self.elem.activeView = view

    @property
    def has_active_view(self) -> bool:
        """Check to see if the Project has an active view set"""
        return bool(self.elem.activeView)

    @property
    def databases(self) -> list[Dataset]:
        """Get a list of all `arcpie.Dataset` objects in the Project.
        By default these are cached on first load
        """
        return self._cached('databases',
            lambda: [
                db.Dataset(str(path))
                for gdb in self.elem.databases
                if (path := gdb.get('databasePath'))
                and Path(str(path)).exists()
            ]
        )

    @property
    def default_database(self) -> Dataset:
        """Get an `arcpie.Dataset` object for the default project database"""
        return db.Dataset(self.elem.defaultGeodatabase)

    @default_database.setter
    def default_database(self, db: Dataset | Path | str) -> None:
        """Can set the default database with a `Dataset`, `Path`, or `str`"""
        if not isinstance(db, Path | str):
            if hasattr(db, 'conn'):
                db = db.conn
            else:
                raise TypeError(f'{type(db)} is not a valid database, must be a Dataset, Path, or str')
        self.elem.defaultGeodatabase = str(db)

    @property
    def metadata(self) -> Metadata:
        """Get the metadata object for the Project"""
        return self.elem.metadata

    @metadata.setter
    def metadata(self, metadata: Metadata) -> None:
        self.elem.metadata = metadata

    @property
    def basemaps(self) -> list[str]:
        """Get the basemap names for the Project"""
        return self.elem.listBasemaps()

    @property
    def color_ramps(self) -> list[mp.ColorRamp]:
        """Get the color ramps in the project"""
        return self.elem.listColorRamps()

    @property
    def index(self) -> dict[str, Any]:
        if 'index' not in self.cache:
            self.cache['index'] = json.loads(self.read_cim_file('Index.json'))
        return self._cached('index')

    @property
    def raw_nodes(self) -> list[APRXNode]:
        return self.index['Nodes']

    @property
    def nodes(self) -> dict[int, ProjectNode]:
        if 'nodes' not in self.cache:
            node_map = dict[int, ProjectNode]()
            for node in self.raw_nodes:
                n_id = node['NodeId']
                n_type = node['NodeType']
                n_path = node['FileName']
                n_children = node['ChildNodeIds']
                cur_node = node_map.setdefault(n_id, cast(ProjectNode, {}))
                cur_node['path'] = n_path
                cur_node['type'] = n_type
                try:
                    cur_node['name'] = json.loads(self.read_cim_file(n_path))['name']
                except Exception:
                    cur_node['name'] = n_path
                if n_children:
                    cur_node['children'] = {int(chld) for chld in n_children.split(',')}
                else:
                    cur_node['children'] = set()
            self.cache['nodes'] = node_map
        return self._cached('nodes')

    def read_cim_file(self, path: Path | str) -> bytes:
        """Open the aprx and read a file inside it (useful for extracting images)."""
        # ZipFile paths need to be in posix (/ delimited) format
        path = Path(path).as_posix()
        with ZipFile(self.path) as zf:
            return zf.read(str(path))

    def update_connection(self, new: str, current: str | None = None, auto_update: bool = True, validate: bool = True, ignore_case: bool = False):
        """Update a connection at the Project level"""
        self.elem.updateConnectionProperties(current, new, auto_update, validate, ignore_case)

    def close_views(self, view_type: Literal['ALL'] | mpt.ViewType = 'ALL', wildcard: str | None = None) -> None:
        """Close open views for a `CURRENT` project

        Args:
            view_type: The type of views to close (default: `ALL`)
            wildcard: A wildcard string that will filter the on the view parent (Map/Report/Layout name)
        Raises:
            PermissionError: If the Project is not `CURRENT`
        """
        if not self.is_current:
            raise PermissionError(f'{self} was not initialized as `CURRENT` and has no views to close')
        if view_type == 'ALL':
            self.elem.closeViews('MAPS_AND_LAYOUTS', wildcard)
            self.elem.closeViews('REPORTS', wildcard)
            self.elem.closeViews('TABLES', wildcard)
        else:
            self.elem.closeViews(view_type, wildcard)

    @overload
    def copy_item(self, item: MapLike, name: str | None = ...) -> Map: ...
    @overload
    def copy_item(self, item: LayoutLike, name: str | None = ...) -> Layout: ...
    @overload
    def copy_item(self, item: ReportLike, name: str | None = ...) -> Report: ...
    def copy_item(
        self,
        item: MapLike | LayoutLike | ReportLike,
        name: str | None = None,
    ) -> Map | Layout | Report:
        """Create a copy of a Map/Layout/Report element in the Project

        Args:
            item: The item to copy
            name: The new name (default: `<original>{n}`)

        Raises:
            TypeError: If the element is not a Map/Layout/Report
        """
        if isinstance(item, Element):
            item = item.elem
        item_type = type(item).__name__
        if item_type not in ('Map', 'Layout', 'Report'):
            raise TypeError(f'{type(item)} cannot be copied!')
        item = self.elem.copyItem(item, name)
        if item_type == 'Map':
            return Map(cast(mpt.Map, item), self)
        elif item_type == 'Layout':
            return Layout(cast(mpt.Layout, item), self)
        elif item_type == 'Report':
            return Report(cast(mpt.Report, item), self)
        else:
            raise Exception('Unreachable')

    def create_layout(self, width: float, height: float, units: mpt.PageUnits, name: str | None = None) -> Layout:
        """Create a new Layout in the Project.

        Args:
            width: The Layout width in `units`
            height: The Layout height in `units`
            units: The Layout page units
            name: An optional name for the Layout (default: `Layout{n}`)
        """
        return Layout(self.elem.createLayout(width, height, units, name), self)

    def create_map(self, name: str | None = None, type: mpt.MapType = 'MAP') -> Map:
        """Create a new Map in the Project.

        Args:
            name: An optional name for the Map (default: `Map{n}`)
            type: The map type to create (default: `MAP`)
        """
        return Map(self.elem.createMap(name, type), self)

    def create_report(
        self,
        units: mpt.PageUnits,
        margin: Literal['NORMAL', 'NARROW', 'MODERATE', 'WIDE'],
        source: LayerLike | TableLike,
        fields: Iterable[ReportFieldConf],
        stats: Iterable[ReportStatConf],
        name: str | None = None,
        template: Literal['ATTR_LIST', 'ATTR_LIST_GROUP', 'BASIC_SUM', 'BASIC_SUM_GROUP', 'PAGE_PER_FEATURE'] = 'ATTR_LIST',
        styling: Literal['BLACK_AND_WHITE', 'COOL_TONES', 'WARM_TONES', 'NO_STYLING'] = 'BLACK_AND_WHITE',
    ) -> Report:
        """Create a new Report in the Project.

        Args:
            units: The Report page units.
            margin: The Report margin width.
            source: The Report data source (Layer or Table).
            fields: A list of dictonaries defining the Report fields.
            stats: A list of dictonaries defining the Report statistics.
            name: An optional name for the Layout. (default: `Report{n}`)
            template: The Report template. (default: `ATTR_LIST`)
            styling: The Report color styling. (default: `BLACK_AND_WHITE`)
        """
        if isinstance(source, Element):
            source = source.elem
        rpt = self.elem.createReport(
            {'units': units, 'margin': margin},
            source,
            list(fields),  # type: ignore (incorrect hint)
            list(stats),  # type: ignore (incorrect hint)
            name,
            template,
            styling
        )
        return Report(rpt, self)

    def create_graphic_element(
        self,
        container: LayoutLike | GroupElementLike,
        geometry: Point | Polyline | Polygon | HasCentroid,
        style_item: StyleItemLike | None = None,
        name: str | None = None,
        lock_aspect_ratio: bool = True,
    ) -> GraphicElement:
        """Create a new GraphicElement in the Project.

        Args:
            container: A Layout or GroupElement to create the GraphicElement in.
            geometry: A Point/Polygon/Polyline or Centroid having object to use as the graphic shape.
            style_item: An optional style to apply to the GraphicElement.
            name: An optional name to give the GraphicElement. (default: `Element{n}`)
            lock_aspect_ratio: Lock the element aspect ratio to prevent skewing. (default: `True`)

        Raises:
            ValueError: If the container element is not a `Layout` or a `GroupElement`
        """
        if isinstance(container, Element):
            container = container.elem
        if isinstance(style_item, Element):
            style_item = style_item.elem
        if not isinstance(geometry, Point | Polyline | Polygon):
            geometry = geometry.centroid
        container_type = type(container).__name__
        if container_type not in ('Layout', 'GroupElement'):
            raise ValueError(
                f'{container_type} cannot be used as container for GraphicElement. '
                'Must be Layout or GroupElement'
            )
        ge = self.elem.createGraphicElement(container, geometry, style_item, name, lock_aspect_ratio)

        if container_type == 'Layout':
            return GraphicElement(ge, Layout(cast(mpt.Layout, container), self))
        elif container_type == 'GroupElement':
            # TODO: Allow Graphic Elements to be parented to GroupElements
            return GraphicElement(ge, None)
        # elif container_type == 'Layer':
        #     # TODO: Allow Graphic Elements to be parented to Layers (Graphic Layers only)
        #     return GraphicElement(ge, None)
        else:
            raise Exception('Unreachable')

    def create_group_element(
        self,
        container: LayoutLike | GroupElementLike,
        elements: Iterable[LayoutElement[mpt.LayoutElement, Any] | mpt.LayoutElement],
        name: str | None,
    ) -> GroupElement:
        """Create a new GroupElement in the Project.

        Args:
            container: A Layout or GroupElement to create the GroupElement in.
            elements: An iterable of LayoutElements to include in the group.
            name: An optional name to give the GroupElement. (default: `Group{n}`)

        Raises:
            ValueError: If the container element is not a `Layout` or a `GroupElement`
        """
        if isinstance(container, Element):
            container = container.elem
        container_type = type(container).__name__
        if container_type not in ('Layout', 'GroupElement'):
            raise ValueError(
                f'GroupElement cannot be created in {container_type}, '
                'must be Layout or GroupElement'
            )
        if not elements:
            raise ValueError('New GroupElement requires elements')
        converted_elements = [
            elem if not isinstance(elem, Element) else elem.elem
            for elem in elements
        ]

        ge = self.elem.createGroupElement(container, converted_elements, name)
        if container_type == 'GroupElement':
            return GroupElement(ge, None)
        elif container_type == 'Layout':
            return GroupElement(ge, Layout(cast(mpt.Layout, container), self))
        else:
            raise Exception('Unreachable')

    def create_picture_element(
        self,
        container: LayoutLike | GroupElementLike,
        geometry: Point | Polyline | Polygon | HasCentroid,
        image: Path | str | bytes,
        image_extension: str | None = None,
        name: str | None = None,
        lock_aspect_ratio: bool = True,
    ) -> PictureElement:
        """Create a new PictureElement in the Project.

        Args:
            container: A Layout or GroupElement to create the PictureElement in.
            geometry: A Point/Polygon/Polyline or Centroid having object to use as the graphic shape.
            image: A file path or raw bytes to use as the image.
            image_extension: If passing raw image bytes, provide an extension (no dot) here. (default: `png`)
            name: An optional name to give the PictureElement. (default: `Element{n}`)
            lock_aspect_ratio: Lock the element aspect ratio to prevent skewing. (default: `True`)

        Raises:
            ValueError: If the container element is not a `Layout` or a `GroupElement`
        """
        if isinstance(container, Element):
            container = container.elem

        container_type = type(container).__name__
        if container_type not in ('Layout', 'GroupElement'):
            raise ValueError(
                f'GroupElement cannot be created in {container_type}, '
                'must be Layout or GroupElement'
            )

        if not isinstance(geometry, Point | Polyline | Polygon):
            geometry = geometry.centroid

        if isinstance(image, bytes):
            with tempfile.TemporaryDirectory() as tmp:
                fl = (Path(tmp) / (name or 'img')).with_suffix(f'.{image_extension or 'png'}')
                fl.write_bytes(image)
                pe = self.elem.createPictureElement(container, geometry, str(fl), name, lock_aspect_ratio)
        else:
            pe = self.elem.createPictureElement(container, geometry, str(image), name, lock_aspect_ratio)

        if container_type == 'GroupElement':
            return PictureElement(pe, None)
        elif container_type == 'Layout':
            return PictureElement(pe, Layout(cast(mpt.Layout, container), self))
        else:
            raise Exception('Unreachable')

    def create_text_element(
        self,
        container: LayoutLike | GroupElementLike,
        geometry: Point | Polyline | Polygon | HasCentroid,
        text_type: mpt.TextType,
        text: str,
        size: float | None = None,
        font: str | None = None,
        style: str | None = None,
        style_item: StyleItemLike | None = None,
        name: str | None = None,
        lock_aspect_ratio: bool = True,
    ) -> TextElement:
        """Create a new TextElement in the Project.

        Args:
            container: A Layout or GroupElement to create the TextElement in.
            geometry: A Point/Polygon/Polyline or Centroid having object to use as the graphic shape.
            text_type: The type of textbox to create.
            text: The text to add to the text element.
            size: The font size (in points) of the text.
            font: The font face to use (must be installed at system level).
            style: The font style to use (e.g. bold, italic). Options depend on selected font face.
            style_item: An optional style item that matches the geometry type provided to the `geometry` arg.
            name: An optional name to give the TextElement. (default: `Element{n}`)
            lock_aspect_ratio: Lock the element aspect ratio to prevent skewing. (default: `True`)

        Raises:
            ValueError: If the container element is not a `Layout` or a `GroupElement`
        """
        if isinstance(container, Element):
            container = container.elem

        container_type = type(container).__name__
        if container_type not in ('Layout', 'GroupElement'):
            raise ValueError(
                f'GroupElement cannot be created in {container_type}, '
                'must be Layout or GroupElement'
            )

        if not isinstance(geometry, Point | Polyline | Polygon):
            geometry = geometry.centroid

        if isinstance(style_item, Element):
            style_item = style_item.elem

        te = self.elem.createTextElement(container, geometry, text_type, text, size, font, style, style_item, name, lock_aspect_ratio)
        if container_type == 'Layout':
            return TextElement(te, Layout(cast(mpt.Layout, container), self))
        elif container_type == 'GroupElement':
            return TextElement(te, None)
        else:
            raise Exception('Unreachable')

    def create_predefined_graphics_element(
        self,
        container: LayoutLike | GroupElementLike,
        geometry: Point | Polyline | Polygon | HasCentroid,
        shape_type: mpt.ShapeType,
        style_item: StyleItemLike | None = None,
        name: str | None = None,
        lock_aspect_ratio: bool = True,
    ) -> GraphicElement:
        """Create a new GraphicElement in the Project using a predefined style.

        Args:
            container: A Layout or GroupElement to create the GraphicElement in.
            geometry: A Point/Polygon/Polyline or Centroid having object to use as the graphic shape.
            shape_type: The predefined shape type to create (see: [docs](https://doc.esri.com/en/arcgis-pro/latest/arcpy/mapping/arcgisproject-class.html#method-createPredefinedGraphicElement)).
            style_item: An optional style item that matches the geometry type provided to the `geometry` arg.
            name: An optional name to give the TextElement. (default: `Element{n}`)
            lock_aspect_ratio: Lock the element aspect ratio to prevent skewing. (default: `True`)

        Raises:
            ValueError: If the container element is not a `Layout` or a `GroupElement`
        """
        if isinstance(container, Element):
            container = container.elem

        container_type = type(container).__name__
        if container_type not in ('Layout', 'GroupElement'):
            raise ValueError(
                f'GroupElement cannot be created in {container_type}, '
                'must be Layout or GroupElement'
            )

        if not isinstance(geometry, Point | Polyline | Polygon):
            geometry = geometry.centroid

        if isinstance(style_item, Element):
            style_item = style_item.elem

        ge = self.elem.createPredefinedGraphicElement(container, geometry, shape_type, style_item, name, lock_aspect_ratio)

        if container_type == 'Layout':
            return GraphicElement(ge, Layout(cast(mpt.Layout, container), self))
        elif container_type == 'GroupElement':
            return GraphicElement(ge, None)
        else:
            raise Exception('Unreachable')

    def save(self) -> None:
        """Save the project.

        Raises:
            ``PermissionError`` if the file is ReadOnly
        """
        if self.is_read_only:
            raise PermissionError(f'{self.name} is read only!')
        self.elem.save()

    def save_as(self, path: Path | str) -> Project:
        """Save a copy of the project"""
        path = Path(path).with_suffix('.aprx')
        self.elem.saveACopy(str(path))
        return type(self)(path)

    def start(self) -> None:
        """Open the project using `os.startfile`"""
        os.startfile(self.path)  # noqa: S606

    def to_directory(self, to: Path | str, *, overwrite: bool = False) -> Path:
        """Unzip the aprx file to a target directory.

        Args:
            to: The folder to unzip the Project into.
            overwrite: Overwrite the contents of the target folder if it exists.

        Raises:
            FileExistsError: If `to` exists and is not an empty directory
        """
        to = Path(to).resolve()
        if to.exists() and not overwrite:
            raise FileExistsError(f'{to} exists and `overwrite` is set to `False`')
        with ZipFile(self.path) as zf:
            zf.extractall(to)
        return to

    def delete(self, *, home_folder: bool = False) -> None:
        """Delete the project.

        Args:
            home_folder: If set, recursively delete the homeFolder (default: `False`)
        """
        if home_folder:
            shutil.rmtree(self.home)
        else:
            self.path.unlink()

    def import_document(self, doc: Path | str | bytes,
                        *,
                        include_layout: bool = True,
                        reuse_existing_maps: bool = True,
                        log: bool = True,
                        extension: str | None = None,
        ) -> Layout | Map | Report:
        """Import a document file into this project using `ArcGISProject.importDocument`.

        Args:
            doc: The path to the document. (`.pagx`, `.mapx`, `.rptx`, `.mxd`, ...[see `importDocument`])
            include_layout: Include layouts with `.mapx` files.
            reuse_existing_maps: Reuse existing maps with `.pagx` files.
            log: Log the import in the ImportLog. (default: `True`)
            extension: If using bytes as the source document, supply the file extension (no dot)
        """
        if isinstance(doc, bytes):
            if not extension:
                raise ValueError('Cannot import bytes data without specified extension')
            with tempfile.TemporaryDirectory() as tmp:
                tmp = (Path(tmp) / '_map').with_suffix(f'.{extension}')
                tmp.write_bytes(doc)
                imported: Any = self.elem.importDocument(
                    str(doc),
                    include_layout=include_layout,
                    reuse_existing_maps=reuse_existing_maps,
                    log_files=log,
                )
        else:
            doc = Path(doc)
            if doc.suffix not in ('.pagx', '.mapx', '.rptx', '.mxd'):
                raise ValueError(
                    f'Document type {doc.suffix} cannot be imported (.pagx, .mapx, .rptx, .mxd)'
                )

            imported: Any = self.elem.importDocument(
                str(doc),
                include_layout=include_layout,
                reuse_existing_maps=reuse_existing_maps,
            )
        imported_type_name = type(imported).__name__
        if imported_type_name == 'Map':
            self.refresh('maps')
            return Map(imported, parent=self)
        if imported_type_name == 'Layout':
            self.refresh('layouts')
            return Layout(imported, parent=self)
        if imported_type_name == 'Report':
            self.refresh('reports')
            return Report(imported, parent=self)
        else:
            raise Exception('Unreachable')

    def import_pagx(self, pagx: Path | str | bytes, *, reuse_existing_maps: bool = False) -> Layout:
        """Import a `.pagx` file. (see: `Project.import_document`)"""
        if not isinstance(pagx, bytes):
            pagx = Path(pagx)
            if pagx.suffix != '.pagx':
                raise ValueError(f'Expected .pagx file, got {pagx.suffix}')
        imported = self.import_document(pagx, reuse_existing_maps=reuse_existing_maps, extension='pagx')
        assert isinstance(imported, Layout)
        return imported

    def import_mapx(self, mapx: Path | str | bytes) -> Map:
        """Import a `.mapx` file. (see: `Project.import_document`)"""
        if not isinstance(mapx, bytes):
            mapx = Path(mapx)
            if mapx.suffix != '.mapx':
                raise ValueError(f'Expected .mapx file, got {mapx.suffix}')
        imported = self.import_document(mapx, extension='mapx')
        assert isinstance(imported, Map)
        return imported

    def import_mxd(self, mxd: Path | str | bytes, *, include_layout: bool = True) -> Map:
        """Import a `.mxd` file. (see: `Project.import_document`)"""
        if not isinstance(mxd, bytes):
            mxd = Path(mxd)
            if mxd.suffix != '.mxd':
                raise ValueError(f'Expected .mxd file, got {mxd.suffix}')
        imported = self.import_document(mxd, include_layout=include_layout, extension='mxd')
        assert isinstance(imported, Map)
        return imported

    def import_rptx(self, rptx: Path | str | bytes) -> Report:
        """Import a `.rptx` file. (see: `Project.import_document`)"""
        if not isinstance(rptx, bytes):
            rptx = Path(rptx)
            if rptx.suffix != '.rptx':
                raise ValueError(f'Expected .rptx file, got {rptx.suffix}')
        imported = self.import_document(rptx, extension='rptx')
        assert isinstance(imported, Report)
        return imported

    def add_database(self, database: Dataset | Path | str, *, default: bool = False):
        current = self.elem.databases
        current.append({'databasePath': str(database), 'isDefaultDatabase': default})
        self.elem.updateDatabases(current)
        self.refresh('databases')

    def __enter__(self) -> Self:
        if not self.is_open:
            self.open()
        if self.is_read_only:
            raise PermissionError(f'{self} is read only')
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        if exc:
            self.close()
            raise exc
        else:
            self.save()
            self.close()

    @classmethod
    def from_directory(cls, directory: Path | str, outfile: Path | str) -> Project:
        """Create an aprx file from a previously unzipped directory (see: `Project.to_directory`)

        Args:
            directory: The source directory (see: `Project.to_directory`).
            outfile: The target `aprx` file.
        """
        directory = Path(directory).resolve()
        outfile = Path(outfile).resolve().with_suffix('.aprx')
        outfile.touch(exist_ok=True)
        with ZipFile(outfile, mode='w', compresslevel=9, compression=ZIP_DEFLATED) as zf:
            for rt, drs, fls in directory.walk():
                for dr in drs:
                    zf.mkdir(dr)
                for fl in fls:
                    fl = rt / fl
                    zf.write(fl, fl.relative_to(directory))
        return cls(outfile)

    @classmethod
    def create(
        cls,
        path: Path | str,
        name: str,
        *,
        home_folder: Path | str | None = None,
        default_database: Path | str | None = None,
        default_toolbox: Path | str | None = None,
        create_parents: bool = True,
    ) -> Self:
        """Create a new Project from scratch.

        Args:
            path: The target location for the project home folder.
            name: The name of the project aprx file.
            home_folder: An optional path to the project home folder. (default: `path.parent`)
            default_database: An optional path to the project default database. (default: `{home_folder}/default.gdb`)
            default_toolbox: An optional path to the project default toolbox. (default: `{home_folder}/default.atbx`)
            create_parents: Create the parent directories for the Project if they don't exist. (default: )
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=create_parents)
        aprx = mp.CreateArcGISProject(
            project_path=str(path),
            project_name=name,
            create_parent_folder=create_parents,
            home_folder=str(home_folder) if home_folder else None,
            default_database=str(default_database) if default_database else None,
            default_toolbox=str(default_toolbox) if default_toolbox else None,
        )
        return cls(aprx.filePath)


class Map(Element[mpt.Map, cim.CIMMap, Project]):
    """mp.Map wrapper that allows for simpler interacton with Maps.

    Example:
    ```python
    >>> my_map = prj.maps['My Map$'].unwrap()
    >>> my_map.layers
    [Layer(Lay 1), Layer(Lay 2), ....]
    >>> lay2 = my_map.layers['2$'].unwrap()
    >>> lay1 = my_map.layers['1$'].unwrap()
    >>> my_map.move_layer(lay1, lay2, 'AFTER')
    >>> my_map.layers
    [Layer(Lay 2), Layer(Lay 1), ....]
    ```
    """

    # Default spatial reference for Maps with unset reference
    _default_reference = SpatialReference('GCS_WGS_1984')
    """Set the fallback reference for Maps that have no assigned spatialReference"""

    @property
    def default_view(self) -> MapView:
        return MapView(self.elem.defaultView, self.parent)

    @property
    def map_type(self) -> mpt.MapType:
        """Get the type of the Map. (`GLOBE`, `SCENE`, `MAP`)"""
        return self.elem.mapType

    @property
    def spatial_reference(self) -> SpatialReference:
        """Get the SpatialReference of the Map (or `Map._default_reference`)."""
        # Get custom ref or return default WGS84/4326
        return self.elem.spatialReference or type(self)._default_reference

    @spatial_reference.setter
    def spatial_reference(self, reference: SpatialReference | int | str) -> None:
        if not isinstance(reference, SpatialReference):
            reference = SpatialReference(reference)
        self.elem.spatialReference = reference

    @property
    def scale(self) -> float:
        """Get the current reference scale of the Map."""
        return self.elem.referenceScale

    @scale.setter
    def scale(self, scale: float) -> None:
        self.elem.referenceScale = scale

    @property
    def camera(self) -> mp.Camera:
        """Get the Map camera."""
        return self.elem.defaultCamera

    @camera.setter
    def camera(self, camera: mp.Camera) -> None:
        self.elem.defaultCamera = camera

    @property
    def units(self) -> str:
        """Get the name of the current Map's units."""
        return self.elem.mapUnits

    @property
    def all_layers(self) -> ElementList[Layer]:
        """Get all Layers in the Map (including GroupLayers and broken Layers).
        By default this is cached on first access.
        """
        return self._cached('all_layers',
            lambda: ElementList(
                Layer(lay, self)
                for lay in self.elem.listLayers()
        ))

    @property
    def layers(self) -> ElementList[Layer]:
        """Get all Layers in the Map (excluding GroupLayers and broken Layers).
        By default this is cached on first access.
        """
        return self._cached('layers',
            lambda: ElementList(
                Layer(lay, self)
                for lay in self.elem.listLayers()
                if lay.supports('ISGROUPLAYER') and not lay.isGroupLayer
                if lay.supports('ISBROKEN') and not lay.isBroken
        ))

    @property
    def broken_layers(self) -> ElementList[Layer]:
        """Get all broken Layers in the Map.
        By default this is cached on first access.
        """
        return self._cached('broken_layers',
            lambda: ElementList(
                Layer(lay, self)
                for lay in self.elem.listLayers()
                if lay.supports('ISBROKEN') and lay.isBroken
        ))

    @property
    def tables(self) -> ElementList[Table]:
        """Get all Tables in the Map.
        By default this is cached on first access.
        """
        return self._cached('tables',
            lambda: ElementList(
                Table(lay, self)
                for lay in self.elem.listTables()
            )
        )

    @property
    def group_layers(self) -> ElementList[GroupLayer]:
        """Get all GroupLayers in the Map.
        By default this is cached on first access.
        """
        return self._cached('group_layers',
            lambda: ElementList(
                GroupLayer(lay, self)
                for lay in self.elem.listLayers()
                if lay.supports('ISGROUPLAYER') and lay.isGroupLayer
        ))

    @property
    def mapx(self) -> bytes:
        """Get the `mapx` file data as bytes."""
        with tempfile.TemporaryDirectory(suffix=self.name) as tmp:
            mapx = Path(tmp) / f'{self.name}.mapx'
            self.elem.exportToMAPX(str(mapx))
            return mapx.read_bytes()

    @property
    def mapx_dict(self) -> dict[str, Any]:
        """Get the `mapx` file data as a Python dictionary."""
        return json.loads(self.mapx)

    @property
    def bkmx(self) -> bytes:
        """Get the `bkmx` file data as bytes."""
        with tempfile.TemporaryDirectory(suffix=self.name) as tmp:
            bkmx = Path(tmp) / f'{self.name}.bkmx'
            self.elem.exportBookmarks(str(bkmx))
            return bkmx.read_bytes()

    @property
    def bkmx_dict(self) -> dict[str, Any]:
        """Get the `bkmx` file data as a Python dictionary."""
        return json.loads(self.bkmx)

    @property
    def metadata(self) -> Metadata:
        """Get the Metadata for the Map."""
        return self.elem.metadata

    @metadata.setter
    def metadata(self, metadata: Metadata) -> None:
        self.elem.metadata = metadata

    @property
    def excluded_from_clipping(self) -> ElementList[Layer]:
        """Get all Layers in the Map that are excluded from clipping."""
        return ElementList(Layer(lay, self) for lay in self.elem.excludedLayersFromClipping)

    @excluded_from_clipping.setter
    def excluded_from_clipping(self, excluded_from_clipping: Iterable[LayerLike]) -> None:
        self.elem.excludedLayersFromClipping = [
            lay.elem if isinstance(lay, Layer) else lay
            for lay in excluded_from_clipping
        ]

    @property
    def transformations(self) -> dict[str, Any]:
        """Get a mapping of all transformations present in the Map."""
        return self.elem.transformations

    @transformations.setter
    def transformations(self, transformations: dict[str, Any]) -> None:
        self.elem.updateTransformations(transformations)

    @property
    def color_model(self) -> mpt.ColorModel:
        """Get the current Map ColorModel. (`CMYK`, `RGB`)"""
        return self.elem.colorModel

    @color_model.setter
    def color_model(self, color_model: mpt.ColorModel) -> None:
        self.elem.setColorModel(color_model)

    def add_basemap(self, basemap: str | LayerLike) -> None:
        """Add a basemap to the Map using the string name from `Project.basemaps` or a Layer object."""
        if isinstance(basemap, str):
            if not self.parent:
                raise ValueError('Adding basemaps by name only supported on Maps with a parent project')
            if self.parent and basemap in self.parent.basemaps:
                return self.elem.addBasemap(basemap)
            else:
                raise ValueError(f'{basemap} is not in {self.parent.basemaps}')
        if isinstance(basemap, Element):
            basemap = basemap.elem
        self.add_layer(basemap, position='BOTTOM')

    def layers_by_type(self, *types: LayerType, invert: bool = False) -> ElementList[Layer]:
        """Matched layers must be of all provided types (or none of provided types if inverted)"""
        types_set = set(types)
        return ElementList(
            lay for lay in self.all_layers
            if (
                lay.types.issuperset(types_set)
                if not invert
                else not lay.types.issuperset(types_set)
            )
        )

    def filter_by_type(self, include: Iterable[LayerType] | None = None, exclude: Iterable[LayerType] | None = None) -> ElementList[Layer]:
        include = set(include or [])
        exclude = set(exclude or [])
        included = self.layers_by_type(*include)
        excluded = self.layers_by_type(*exclude, invert=True)
        return ElementList(lay for lay in included if lay not in excluded)

    @overload
    def add_layer(self, layer: LayerLike,
                  *,
                  before: LayerLike | None = ...,
                  after: LayerLike | None = ...,
                  position: mpt.AddPosition = ...,
                  group: GroupLayerLike | None = ...) -> Layer: ...
    @overload
    def add_layer(self, layer: GroupLayer,
                  *,
                  before: LayerLike | None = ...,
                  after: LayerLike | None = ...,
                  position: mpt.AddPosition = ...,
                  group: GroupLayerLike | None = ...) -> GroupLayer: ...
    def add_layer(self, layer: LayerLike | GroupLayerLike,
                  *,
                  before: LayerLike | None = None,
                  after: LayerLike | None = None,
                  position: mpt.AddPosition = 'AUTO_ARRANGE',
                  group: GroupLayerLike | None = None) -> Layer | GroupLayer:
        """Add a layer/table object to the map at the provided position.

        Args:
            layer: The layer or table object to add (either arcpie or arcpy versions will work)
            before: Insert the layer before this layer
            after: Insert the layer after this layer
            position: The position in the contents pane to add the layer to (default: `AUTO_ARRANGE`)
            group: An optional GroupLayer to add the new layer to

        Note:
            `before`, `after`, and `group` + `position` are exclusive.
            `before` takes precedence over all other arguments followed by `after`, then `position` and `group`
        """
        is_group = isinstance(layer, GroupLayer)
        layer = layer.elem if isinstance(layer, Element) else layer
        group = group.elem if isinstance(group, Element) else group
        before = before.elem if isinstance(before, Element) else before
        after = after.elem if isinstance(after, Element) else after

        if before:
            elem = self.elem.insertLayer(before, layer, 'BEFORE')
        elif after:
            elem = self.elem.insertLayer(after, layer, 'AFTER')
        elif group:
            elem = self.elem.addLayerToGroup(group, layer, position)[0]
        else:
            elem = self.elem.addLayer(layer, position)[0]

        if type(elem).__name__ != 'Layer':
            raise ValueError(f'Expected Layer object, got {type(elem)}. use `add_table` to add Tables.')

        if group is not None:
            self.remove(layer)

        if not is_group:
            self.refresh('layers')
            return Layer(cast(mpt.Layer, elem), self)
        else:
            self.refresh('group_layers')
            return GroupLayer(cast(mpt.Layer, elem), self)

    def move_layer(self, layer: LayerLike | GroupLayerLike, reference: LayerLike | GroupLayerLike, position: mpt.MovePosition = 'BEFORE') -> None:
        """Move a layer to a position (`BEFORE`/`AFTER`) relative to the reference Layer."""
        layer = layer.elem if isinstance(layer, Element) else layer
        reference = reference.elem if isinstance(reference, Element) else reference
        self.elem.moveLayer(reference, layer, position)

    def add_table(self, table: TableLike,
                  *,
                  group: GroupLayerLike | None = None) -> Table:
        """Add a Table to the Map.

        Args:
            table: The table to add.
            group: An optional GroupLayer to add the Table to.
        """
        if isinstance(table, Table):
            table = table.elem
        if isinstance(group, GroupLayer):
            group = group.elem
        elem = (
            self.elem.addTableToGroup(group, table)
            if group is not None
            else self.elem.addTable(table)
        )
        if not isinstance(elem, mp.Table):
            raise ValueError(f'Expected Table object, got {type(elem)}. use `add_layer` to add Layers.')
        self.refresh('tables')
        return Table(elem, self)

    def remove(self, child: LayerLike | TableLike | BookmarkLike) -> None:
        """Remove a Map Element. (Layer/Table/Bookmark)"""
        child = child.elem if isinstance(child, Element) else child
        child_type = type(child).__name__
        {
            'Layer': self.elem.removeLayer,
            'Table': self.elem.removeTable,
            'Bookmark': self.elem.removeBookmark,
        }.get(child_type, _noop)(child)  # type: ignore
        self.refresh(child_type.lower() + 's')

    def create_group(self, name: str,
                     *,
                     parent: GroupLayerLike | None = None) -> GroupLayer:
        """Create a new GroupLayer in the Map.

        Args:
            name: The name of the new GroupLayer.
            parent: An optional parent group for the new layer.
        """
        if isinstance(parent, GroupLayer):
            parent = parent.elem
        return GroupLayer(self.elem.createGroupLayer(name, parent), self)

    def create_graphics_layer(self, name: str) -> Layer:
        """Create a GraphicsLayer in the Map with the provided name"""
        return Layer(self.elem.createGraphicsLayer(name), self)

    def clear_selection(self) -> None:
        """Clear all selections in the Map."""
        self.elem.clearSelection()

    def copy_bookmark(self, bookmark: BookmarkLike, name: str | None = None) -> Bookmark:
        """Make a copy of the Bookmark with an optional new name."""
        bookmark = bookmark.elem if isinstance(bookmark, Element) else bookmark
        return Bookmark(self.elem.copyBookmark(bookmark, name), self)

    def update_connection(self, new: Path | str, current: Path | str | None = None, auto_update: bool = True, validate: bool = True, ignore_case: bool = False) -> None:
        """Update data connections in the Map.

        Args:
            new: The new connection path.
            current: The current connection path.
            auto_update: Update all existing joins and relates.
            validate: Only complete the connection change if the new path is valid.
            ignore_case: Ignore case in queries.
        """
        self.elem.updateConnectionProperties(str(current), str(new), auto_update, validate, ignore_case)
        self.refresh()

    def clip_to(self, layer: LayerLike, selected: bool = False) -> None:
        """Clip all layers in the map to the footprint of the input layer

        Args:
            layer: The Layer to clip to.
            selected: Clip only to the selected features. (default: `False`)
        """
        if isinstance(layer, Layer):
            layer = layer.elem
        self.elem.clipLayers(layer, selection='SELECTED' if selected else 'ALL')

    @overload
    def add_data(self, path: fc.FeatureClass, service_type: mpt.WebServiceType = ..., group: GroupLayerLike | None = ..., **params: Any) -> Layer: ...  # type: ignore
    @overload
    def add_data(self, path: fc.Table, service_type: mpt.WebServiceType = ..., group: GroupLayerLike | None = ..., **params: Any) -> Table: ...
    @overload
    def add_data(self, path: Path | str, service_type: mpt.WebServiceType = ..., group: GroupLayerLike | None = ..., **params: Any) -> Layer | Table: ...
    def add_data(self,
                 path: fc.FeatureClass | fc.Table | Path | str,
                 service_type: mpt.WebServiceType = 'AUTOMATIC',
                 group: GroupLayerLike | None = None,
                 **params: Any,
        ) -> Layer | Table:
        """Add data to the map from a path or URL.

        Args:
            path: A filepath or URL for the data element
            service_type: An optional agument for web service type (default: `AUTOMATIC`)
            **params: Additional keyword parameters passed to the webservice (optional)
        """
        elem = self.elem.addDataFromPath(str(path), web_service_type=service_type, custom_parameters=params or None)
        group = group.elem if isinstance(group, GroupLayer) else group

        elem_name = type(elem).__name__
        if elem_name == 'Layer':
            lay = Layer(cast(mpt.Layer, elem), self)
            if group is not None:
                group_lay = self.add_layer(lay, group=group)
                lay.delete()
                return group_lay
            self.refresh('layers')
            return lay
        if elem_name == 'Table':
            tbl = Table(cast(mpt.Table, elem), self)
            if group is not None:
                group_tbl = self.add_table(tbl, group=group)
                tbl.delete()
                return group_tbl
            self.refresh('tables')
            return tbl

        # Unreachable ? (at least not documented as reachable...)
        raise ValueError(f'Something went wrong got {type(elem)} but expected Layer or Table')

    def export_mapx(self, outfile: Path | str) -> Path:
        """Export the Map's `mapx` representation to a file."""
        outfile = Path(outfile).with_suffix('.mapx')
        with outfile.open('wb') as mapx:
            mapx.write(self.mapx)
        return outfile

    def export_bkmx(self, outfile: Path | str) -> Path:
        """Export the Map's `bkmx` data to a file."""
        outfile = Path(outfile).with_suffix('.bkmx')
        with outfile.open('wb') as bkmx:
            bkmx.write(self.bkmx)
        return outfile

    def import_bkmx(self, bkmx: Path | str | bytes) -> None:
        """Import Bookmarks from a `bkmx` file."""
        if isinstance(bkmx, bytes):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = (Path(tmp) / '_bookmarks').with_suffix('.bkmx')
                tmp.write_bytes(bkmx)
                self.elem.importBookmarks(str(tmp))
        else:
            self.elem.importBookmarks(str(bkmx))

    def filter(self, pred: Callable[[Layer | Table | GroupLayer], bool]) -> ElementList[Layer | Table | GroupLayer]:
        return self.layers.filter(pred) + self.tables.filter(pred) + self.group_layers.filter(pred)

    def filter_layers(self, pred: Callable[[Layer], bool]) -> ElementList[Layer]:
        """Filter Layers using a predicate function."""
        return self.layers.filter(pred)

    def filter_tables(self, pred: Callable[[Table], bool]) -> ElementList[Table]:
        """Filter Tables using a predicate function."""
        return self.tables.filter(pred)

    def filter_groups(self, pred: Callable[[GroupLayer], bool]) -> ElementList[GroupLayer]:
        """Filter Layers using a predicate function."""
        return self.group_layers.filter(pred)

    def open_view(self) -> None:
        self.elem.openView()


class MapView(Element[mpt.MapView, cim.CIMMapView, Project]):

    @property
    def camera(self) -> mp.Camera:
        """Get the camera of the MapView"""
        return self.elem.camera

    @camera.setter
    def camera(self, camera: mp.Camera) -> None:
        self.elem.camera = camera

    @property
    def extent(self) -> Extent:
        """Get the current extent of the MapView camera"""
        return self.camera.getExtent()

    @property
    def map(self) -> Map:
        """Get the Map that the MapView is associated with (parent is actually Project)"""
        return Map(self.elem.map, self.parent)

    @overload
    def export(self,  # type: ignore (No Overlap?)
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: None = None,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> bytes: ...
    @overload
    def export(self,
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: Path | str = ...,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> Path: ...
    def export(self,
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: Path | str | None = None,
               antialiasing: mpt.Antialiasing | None = None,
        ) -> Path | bytes:
        """Export the MapView.

        Args:
            format: A Format object or a string format (string option will use defaults)
            outfile: An optional output file location to use (will override format filePath)
            antialiasing: Antialiasing options for the output file

        Returns:
            Path: If outfile is set
            bytes: If outfile is unset
        """
        display = None
        outfile = Path(outfile) if outfile else None
        if antialiasing:
            display = cast(mpt.DisplayOptions, mp.CreateExportOptions('DISPLAY'))
            display.setAntialiasing(antialiasing)

        if isinstance(format, fmts.Format):
            format = format.fmt

        if type(format).__name__.endswith('Format'):
            suffix = type(format).__name__[:3].lower()
            format = cast(mpt.ExportFormat, format)

        elif isinstance(format, str):
            suffix = format[:3].lower()
            format = mp.CreateExportFormat(format)

        else:
            raise ValueError(f'Unknown format {type(format)} : {format}')

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / f'{self.name}.{suffix}'
            format.filePath = str(out)
            self.elem.export(format, display_options=display)
            data = out.read_bytes()
        if outfile:
            outfile.parent.mkdir(exist_ok=True, parents=True)
            outfile.with_suffix(suffix).write_bytes(data)
            return outfile
        else:
            return data

    def create_bookmark(self, name: str | None = None, description: str | None = None) -> Bookmark:
        """Create a new Bookmark from the current MapView

        Args:
            name: An optional name of the Bookmark. (default: `Bookmark{n}`)
            description: An optional description for the Bookmark.
        """
        return Bookmark(self.elem.createBookmark(name, description), self.map)

    def layer_extent(self, layer: LayerLike, selected: bool = True, symbolized: bool = True) -> Extent:
        """Get the Extent for a Layer in the MapView.

        Args:
            layer: The layer to get an Extent for.
            selected: Get the extent for only the selected features in the View.
            symbolized: Consider symbology when getting the Extent.
        """
        layer = layer.elem if isinstance(layer, Element) else layer
        return self.elem.getLayerExtent(layer, selected, symbolized)

    def pan_to(self, extent: LayerLike | Polygon | Extent) -> None:
        """Pan the MapView to an Extent/Layer/Polygon.

        Args:
            extent: A Layer, Polygon, or Extent to pan to.
        """
        if isinstance(extent, Extent):
            extent = extent
        elif isinstance(extent, Polygon):
            extent = extent.extent
        else:
            extent = self.layer_extent(extent)
        self.elem.panToExtent(extent)

    def zoom_all(self, selected: bool = True, symbolized: bool = True) -> None:
        """Zoom the MapView to all layers.

        Args:
            selected: Zoom to only the selected features.
            symbolized: Consider symbology when zooming.
        """
        self.elem.zoomToAllLayers(selected, symbolized)

    def zoom_to(self, elem: LayerLike | BookmarkLike | Extent | HasExtent) -> None:
        """Zoom the MapView to a Layer or Bookmark.

        Args:
            elem: The Layer or Bookmark to zoom to.
        """
        if isinstance(elem, HasExtent):
            self.camera.setExtent(elem.extent)
            return
        elif isinstance(elem, Element):
            elem = elem.elem

        elem_type = type(elem).__name__
        if elem_type == 'Bookmark':
            self.elem.zoomToBookmark(cast(mpt.Bookmark, elem))
        elif elem_type == 'Layer':
            elem = cast(mpt.Layer, elem)
            self.camera.setExtent(self.layer_extent(elem))


class GroupLayer(Element[mpt.Layer, cim.CIMGroupLayer, "Map | GroupLayer"]):

    @property
    def spatial_reference(self) -> SpatialReference:
        if not self.parent:
            raise AttributeError(f'{self} has no associated map, and no spatial reference')
        if isinstance(self.parent, GroupLayer):
            return self.parent.spatial_reference
        return self.parent.spatial_reference

    @property
    def group_type(self) -> mpt.GroupType:
        return self.elem.groupType

    @property
    def layers(self) -> ElementList[Layer]:
        return self._cached('layers',
            lambda: ElementList(
                Layer(lay, self)
                for lay in self.elem.listLayers()
                if not lay.isGroupLayer
            )
        )

    @property
    def tables(self) -> ElementList[Table]:
        return self._cached('tables',
            lambda: ElementList(
                Table(lay, self)
                for lay in self.elem.listTables()
            )
        )

    @property
    def group_layers(self) -> ElementList[GroupLayer]:
        return self._cached('group_layers', lambda: ElementList(
            GroupLayer(lay, self.parent)
            for lay in self.elem.listLayers()
            if lay.isGroupLayer
        )
    )

    @property
    def lyrx(self) -> dict[str, Any]:
        lyrx = dict[str, Any]()
        lyrx['type'] = 'CIMLayerDocument'
        lyrx['layers'] = [self.uri]
        lyrx['layerDefinitions'] = [self.cim_dict] + [lay.cim_dict for lay in self.layers + self.group_layers]
        lyrx['tableDefinitions'] = [tab.cim_dict for tab in self.tables]
        return lyrx

    def add_layer(self, layer: LayerLike, position: mpt.AddPosition = 'AUTO_ARRANGE', *, copy: bool = False) -> Layer:
        parent = self.parent
        layer = layer.elem if isinstance(layer, Element) else layer
        while not isinstance(parent, Map | None):
            parent = parent.parent
        if parent is None:
            raise ValueError(f'{self} is unbound and cannot add new layers')
        new_lay = parent.add_layer(layer, position=position, group=self)
        if copy:
            parent.refresh('layers')
            return new_lay
        else:
            parent.move_layer(layer, new_lay.elem, 'AFTER')
            parent.remove(new_lay)
            self.refresh('layers')
            return Layer(layer, self)

    def add_table(self, table: TableLike) -> Table:
        parent = self.parent
        table = table.elem if isinstance(table, Element) else table
        while not isinstance(parent, Map | None):
            parent = parent.parent
        if parent is None:
            raise ValueError(f'{self} is unbound and cannot add new layers')
        new_table = parent.add_table(table, group=self)
        parent.refresh('tables')
        return new_table

    @overload
    def add_data(self, path: fc.FeatureClass, service_type: mpt.WebServiceType = ..., **params: Any) -> Layer: ...  # type: ignore
    @overload
    def add_data(self, path: fc.Table, service_type: mpt.WebServiceType = ..., **params: Any) -> Table: ...
    @overload
    def add_data(self, path: Path | str, service_type: mpt.WebServiceType = ..., **params: Any) -> Layer | Table: ...
    def add_data(self, path: fc.FeatureClass | fc.Table | Path | str, service_type: mpt.WebServiceType = 'AUTOMATIC', **params: Any) -> Layer | Table:
        """Add data to the Group from a path or URL.

        Args:
            path: A filepath or URL for the data element
            service_type: An optional agument for web service type (default: `AUTOMATIC`)
            **params: Additional keyword parameters passed to the webservice (optional)
        """
        parent = self.parent
        while parent and not isinstance(parent, Map):
            parent = parent.parent
        if not parent:
            raise ValueError(f'{self} has no associated Map')

        item = parent.add_data(path, service_type, **params)
        if isinstance(item, Layer):
            item = parent.add_layer(item, group=self)
            self.refresh('layers')
            return item
        else:
            tbl = parent.add_table(item, group=self)
            self.refresh('tables')
            return tbl

    def export_lyrx(self, outdir: Path | str, *, name: str | None = None, indent: int = 2) -> Path:
        outdir = Path(outdir)
        name = name or self.name
        outdir = outdir / name
        outdir = outdir.with_suffix('.lyrx')
        outdir.parent.mkdir(exist_ok=True, parents=True)
        outdir.write_text(json.dumps(self.lyrx, indent=indent))
        return outdir


# Alternate type names so layer types can be given as a set
LayerType = Literal[
    '3d',
    'basemap',
    'broken',
    'feature',
    'graphics',
    'group',
    'network-analyst',
    'network-dataset',
    'parcel-fabric',
    'raster',
    'scene',
    'time-enabled',
    'topology',
    'web',
    # Special Types
    'annotation-layer',
    'annotation-sublayer',
    'dimension-layer',
    'terrain-layer',
    'raster-catalog-layer',
]
LayerTypes: tuple[LayerType, ...] = LayerType.__args__

LayerProperty = Literal[
    'brightness',
    'connectionProperties',
    'contrast',
    'dataSource',
    'definitionQuery',
    'elevation',
    'is3DLayer',
    'isBasemapLayer',
    'isBroken',
    'isFeatureLayer',
    'isGraphicsLayer',
    'isGroupLayer',
    'isNetworkAnalystLayer',
    'isNetworkDatasetLayer',
    'isParcelFabricLayer',
    'isRasterLayer',
    'isSceneLayer',
    'isTimeEnabled',
    'isWebLayer',
    'longName',
    'maxThreshold',
    'metadata',
    'minThreshold',
    'name',
    'showLabels',
    'symbology',
    'transparency',
    'time',
    'URI',
    'visible',
]
LayerProps = frozenset[LayerProperty](LayerProperty.__args__)
_PropMap: dict[str, LayerProperty] = {p.lower(): p for p in LayerProps}


class LayerConnectionConf(TypedDict):
    dataset: str
    workspace_factory: str
    connection_info: dict[str, str | bool]


# Note: If a property is requested, but not supported
# it will be sentinel `Layer.NotSupported`
class AllLayerProps(TypedDict):
    brightness: int
    connectionProperties: LayerConnectionConf
    contrast: int
    dataSource: str
    definitionQuery: str
    elevation: mpt.LayerElevation
    is3DLayer: bool
    isBasemapLayer: bool
    isBroken: bool
    isFeatureLayer: bool
    isGraphicsLayer: bool
    isGroupLayer: bool
    isNetworkAnalystLayer: bool
    isNetworkDatasetLayer: bool
    isParcelFabricLayer: bool
    isRasterLayer: bool
    isSceneLayer: bool
    isTimeEnabled: bool
    isWebLayer: bool
    longName: str
    maxThreshold: float
    metadata: Metadata
    minThreshold: float
    name: str
    showLabels: bool
    symbology: Symbology
    transparency: int
    time: mpt.LayerTime
    URI: str
    visible: bool


# Alternate Props that set all keys to NotRequired
class SomeLayerProps(TypedDict, total=False):
    brightness: int
    connectionProperties: LayerConnectionConf
    contrast: int
    dataSource: str
    definitionQuery: str
    elevation: mpt.LayerElevation
    is3DLayer: bool
    isBasemapLayer: bool
    isBroken: bool
    isFeatureLayer: bool
    isGraphicsLayer: bool
    isGroupLayer: bool
    isNetworkAnalystLayer: bool
    isNetworkDatasetLayer: bool
    isParcelFabricLayer: bool
    isRasterLayer: bool
    isSceneLayer: bool
    isTimeEnabled: bool
    isWebLayer: bool
    longName: str
    maxThreshold: float
    metadata: Metadata
    minThreshold: float
    name: str
    showLabels: bool
    symbology: Symbology
    transparency: int
    time: mpt.LayerTime
    URI: str
    visible: bool


class DefinitionQuery(TypedDict):
    name: str
    sql: str
    isActive: bool


class _PropNotSupported:
    def __repr__(self) -> str:
        return 'Property Not Supported'

    def __bool__(self):
        return False


NotSupported = _PropNotSupported()


class PageQuery(NamedTuple):
    fieldName: str
    match: bool


class Layer(Element[mpt.Layer, cim.CIMBaseLayer, Map | GroupLayer]):

    # A different sentinel might be needed here
    NotSupported = NotSupported
    """A sentinel for signaling that a layer property is not available for the given layer
    Use with identity checks:
    >>> if lay.props['brightness'] is Layer.NotSupported: ... # handle
    """

    @property
    def types(self) -> set[LayerType]:
        """Get a set of string identifiers for the Layer Type."""
        types = set[LayerType]()
        elem = self.elem
        if getattr(elem, 'is3DLayer', False):
            types.add('3d')
        if getattr(elem, 'isBasemapLayer', False):
            types.add('basemap')
        if getattr(elem, 'isBroken', False):
            types.add('broken')
        if getattr(elem, 'isFeatureLayer', False):
            types.add('feature')
        if getattr(elem, 'isGraphicsLayer', False):
            types.add('graphics')
        if getattr(elem, 'isGroupLayer', False):
            types.add('group')
        if getattr(elem, 'isNetworkAnalystLayer', False):
            types.add('network-analyst')
        if getattr(elem, 'isNetworkDatasetLayer', False):
            types.add('network-dataset')
        if getattr(elem, 'isParcelFabricLayer', False):
            types.add('parcel-fabric')
        if getattr(elem, 'isRasterLayer', False):
            types.add('raster')
        if getattr(elem, 'isSceneLayer', False):
            types.add('scene')
        if getattr(elem, 'isTimeEnabled', False):
            types.add('time-enabled')
        if getattr(elem, 'isTopologyLayer', False):
            types.add('topology')
        if getattr(elem, 'isWebLayer', False):
            types.add('web')

        # Special Cases (TODO: Fill out more of these as encountered)
        if not types:
            try:
                cim_type = self.cim_type_name
                if cim_type == 'CIMAnnotationLayer':
                    types.add('annotation-layer')
                if cim_type == 'CIMAnnotationSubLayer':
                    types.add('annotation-sublayer')
                if cim_type == 'CIMDimensionLayer':
                    types.add('dimension-layer')
                if cim_type == 'CIMProfileTerrain':
                    types.add('terrain-layer')
                if cim_type == 'CIMRasterCatalogLayer':
                    types.add('raster-catalog-layer')
                if not types:
                    raise TypeError(f'{self} has unknown layer type: {cim_type}')
            except json.JSONDecodeError:
                return {'broken', }
        return types

    # Since layer properties cannot be determined until runtime
    # we need to wrap property access. Any overrides should use this
    # interface to allow silent ignoring of unsupported property reads/writes

    @property
    def supported(self) -> frozenset[str]:
        """Get the suported properties of the Layer (see: `LayerProperty`)"""
        props = set(LayerProps)
        if not self.elem.supports('NAME'):
            # Broken web group children will crash Arc
            # if you inspect time support (or the time attribute)
            # name is also not supported on these so this is a pretty
            # safe bet
            props.remove('time')
        return self._cached('supported',
            lambda: frozenset({
                prop for prop in props
                if self.elem.supports(cast(mpt.LayerProperty, prop.upper()))
            }))

    @property
    def unsupported(self) -> frozenset[str]:
        """Get the unsupported properties of the Layer (see: `LayerProperty`)"""
        return frozenset({
            prop for prop in LayerProps
            if prop not in self.supported
        })

    @property
    def props(self) -> AllLayerProps:
        """Get all properties of the layer with unsupported properties being set to `Layer.NotSupported`"""
        return cast(AllLayerProps, self.get_props(*LayerProps))

    @props.setter
    def props(self, props: SomeLayerProps) -> None:
        self.set_props(*props)

    def has(self, prop: LayerProperty) -> bool:
        """Alias for `elem.supports` that accepts lowercased `LayerProperties`"""
        return self.elem.supports(cast(mpt.LayerProperty, prop.upper()))

    def get_props(self, *props: LayerProperty) -> SomeLayerProps:
        """Get requested layer properties.

        Note:
            `*props` is case insensitive (underscores are replaced with empty strings)
            (e.g. `dataSource` == `DATASOURCE` == `datasource` == `data_source`).
            The returned dictionary will use the supplied case.
            (e.g. `layer.get_props('DATASOURCE')` -> `{'DATASOURCE': ...}`)
            if a property is not supported, `Layer.NotSupported` is set as the value.
            (e.g `layer.get_props('brightness') -> {'brightness': Layer.NotSupported}`)
        Raises:
            ValueError: If a property is not a layer property
        """
        p_map = _PropMap
        invalid = [
            p_map[p] for p in
            {p.lower().replace('_', '') for p in props}.difference(p_map.keys())
        ]
        if invalid:
            raise ValueError(
                f'{invalid} are not valid layer properties: {list(LayerProps)}'
            )
        props = props or cast(tuple[LayerProperty], tuple(self.supported))
        caseless_props = [p.lower().replace('_', '') for p in props]
        return cast(SomeLayerProps, {
            prop: (
                getattr(self.elem, prop_name)
                if prop_name in self.supported
                else type(self).NotSupported
            )
            for prop, c_prop in zip(props, caseless_props, strict=True)
            if (prop_name := p_map.get(c_prop))
        })

    def get_prop(self, prop: LayerProperty) -> Any:
        """Get a single property or `Layer.NotSupported`."""
        if prop not in LayerProps:
            return type(self).NotSupported
        return getattr(self.elem, prop, type(self).NotSupported)

    def set_props(self, **props: Unpack[SomeLayerProps]) -> None:
        """Set the provided properties of the Layer."""
        p_map = _PropMap
        for prop, val in props.items():
            prop = p_map.get(prop.lower())
            if prop in self.supported:
                setattr(self.elem, prop, val)

    def set_prop(self, prop: LayerProperty, val: Any) -> None:
        """Set a single property of the layer (if it is supported)."""
        if self.get_prop(prop) is type(self).NotSupported:
            return
        setattr(self.elem, prop, val)

    @property
    def page_query(self) -> PageQuery:
        """Get the active PageQuery of the layer."""
        return cast(PageQuery, self.elem.pageQuery)

    @page_query.setter
    def page_query(self, pq: PageQuery | tuple[str, bool] | None) -> None:
        self.elem.setPageQuery(*pq or ('None', True))

    @property
    def is_geographic(self) -> bool:
        """Determine if the Layer is projected by comparing the SR GCS to the SR."""
        ref = self.spatial_reference
        return ref == ref.GCS

    @property
    def is_projected_on_fly(self) -> bool:
        """Determine if the layer is projected on the fly by comparing datasource SR to layer SR."""
        return (
            (parent := self.parent) is not None and self.has_feature_class
            and (
                self.feature_class.spatial_reference != parent.spatial_reference
            )
        )

    @property
    def spatial_reference(self) -> SpatialReference:
        """Get the SR of the Layer by checking the associated FeatureClass."""
        if self.parent:
            return self.parent.spatial_reference
        elif self.has_feature_class:
            return self.feature_class.spatial_reference
        else:
            # Assume default WGS84 reference
            return SpatialReference(4326)

    @property
    def source_data(self) -> fc.FeatureClass:
        """Alias for `feature_class` that matches `Table` for use in duck typed loops."""
        return self.feature_class

    @property
    def feature_class(self) -> fc.FeatureClass:
        """Get the associated FeatureClass object for the layer.

        validating connections is slow, so use `has_feature_class` if you need a guard
        """
        return fc.FeatureClass.from_layer(self.elem)

    @property
    def has_feature_class(self) -> bool:
        """Check to see if the Layer has a valid `FeatureClass` association."""
        return 'feature' in self.types and fc.FeatureClass.from_layer(self.elem).exists

    @property
    def relative_cim_dict(self) -> dict[str, Any]:
        """A copy of the CIM data with dataConnections made relative to the project."""
        cim_dict = self.cim_dict
        # Convert absolute paths to relative databases into relative paths
        # This allows sharing layers b/w projects with the same structure
        if (
            (ft := cim_dict.get('featureTable'))
            and (conn := cast(dict[str, Any], ft.get('dataConnection')))
            and (ws_conn := str(conn.get('workspaceConnectionString')))
            and conn.get('workspaceFactory') == 'FileGDB'
            and (feature_class := self.feature_class)
        ):
            if not ws_conn.startswith('DATABASE='):
                return self.cim_dict
            cur_path = Path(ws_conn[9:])
            database = Path(feature_class.path).parent
            if cur_path.is_absolute() and cur_path.is_relative_to(database.parent):
                conn['workspaceConnectionString'] = f'DATABASE={cur_path.relative_to(database.parent)}'
        return cim_dict

    @property
    def lyrx(self) -> dict[str, Any]:
        """A dictionary representing the lyrx definition of the Layer."""
        lyrx = dict[str, Any]()
        lyrx['type'] = 'CIMLayerDocument'
        lyrx['layers'] = [self.uri]
        lyrx['layerDefinitions'] = [self.cim_dict]
        return lyrx

    @property
    def relative_lyrx(self) -> dict[str, Any]:
        """A copy of the lyrx with relative pathing for dataConnections."""
        lyrx = dict[str, Any]()
        lyrx['type'] = 'CIMLayerDocument'
        lyrx['layers'] = [self.uri]
        lyrx['layerDefinitions'] = [self.relative_cim_dict]
        return lyrx

    @property
    def visible(self) -> bool:
        """Get the visibility state of the Layer (or `Layer.NotSupported`)."""
        return self.get_prop('visible')

    @visible.setter
    def visible(self, visible: bool) -> None:
        if self.visible is type(self).NotSupported:
            return
        self.elem.visible = visible

    @property
    def data_source(self) -> str:
        """Get the string representation of the Layer datasource (or `Layer.NotSupported`)."""
        return self.get_prop('dataSource')

    @data_source.setter
    def data_source(self, source: str | fc.FeatureClass) -> None:
        if self.data_source is type(self).NotSupported:
            return
        if isinstance(source, fc.FeatureClass):
            source = source.path
        self.update_connection(source)

    @property
    def definition_queries(self) -> list[DefinitionQuery]:
        """Get the DesinitonQueries for the Layer (empty list if no queries or not suppported)."""
        if self.definition_query is type(self).NotSupported:
            return []
        return cast(list[DefinitionQuery], self.elem.listDefinitionQueries())

    @definition_queries.setter
    def definition_queries(self, queries: Iterable[DefinitionQuery]) -> None:
        if self.definition_query is type(self).NotSupported:
            return
        queries = list(queries)
        if total_active := sum(1 for q in queries if q.get('isActive')) > 1:
            raise ValueError(f'{total_active} queries are set to active (max: 1)')
        self.elem.updateDefinitionQueries(cast(list[dict[str, str | bool]], queries))

    @property
    def definition_query(self):
        """Get the active DefinitionQuery string for the Layer (or `Layer.NotSupported`)."""
        return self.get_prop('definitionQuery')

    @definition_query.setter
    def definition_query(self, query: str | None) -> None:
        self.set_props(definitionQuery=query or '')

    @property
    def selection(self) -> set[int]:
        """Get a set of all selected feaature OIDs."""
        return set(self.feature_class['OID@'])

    @selection.setter
    def selection(self, selection: Iterable[int]) -> None:
        self.elem.setSelectionSet(list(set(selection)))

    @property
    def symbology(self) -> Symbology:
        """Get the layer symbology (or `Layer.NotSupported`)."""
        return self.get_prop('symbology')

    @symbology.setter
    def symbology(self, symbology: Symbology) -> None:
        self.set_props(symbology=symbology)

    @property
    def cim_symbology(self) -> cim.CIMSymbolizers.CIMRenderer:
        """Get the symbology CIM object (or `Layer.NotSupported`)."""
        if self.symbology is type(self).NotSupported:  # type: ignore
            return type(self).NotSupported  # type: ignore
        return cast('cim.CIMSymbolizers.CIMRenderer', self.elem.getSymbologyDefinition('V3'))

    @cim_symbology.setter
    def cim_symbology(self, cim_symbology: cim.CIMSymbolizers.CIMRenderer) -> None:
        if self.symbology is type(self).NotSupported:  # type: ignore
            return
        self.elem.setSymbologyDefinition(cast(str, cim_symbology))

    @property
    def cim_symbology_type(self) -> type[cim.CIMSymbolizers.CIMRenderer]:
        """Get the type of the CIM Symbology (or `type(Layer.NotSupported)`)"""
        return type(self.cim_symbology)

    def delete(self) -> None:
        """Delete the Layer."""
        parent = self.parent
        while not isinstance(parent, Map | None):
            parent = parent.parent
        if parent is None:
            raise AttributeError(f'{self} has no associated map and no spatial reference')
        parent.remove(self)

    @contextmanager
    def query_as(self, query: str | None):
        """Context manager for setting the active DefinitionQuery string.

        Example:
        ```python
        >>> print(len(lay))
        250
        >>> with lay.query_as("STATUS = 'ACTIVE'"):
        ...     print(len(lay))
        125
        ```
        """
        cur = self.definition_query
        try:
            self.definition_query = query
            yield self
        finally:
            self.definition_query = cur

    def __len__(self) -> int:
        if self.has_feature_class:
            return len(self.feature_class)
        return 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.has_feature_class:
            return iter(self.feature_class)
        return iter(())

    def __bool__(self) -> Literal[True]:
        # Override __bool__ to prevent __iter__/__len__ from making a layer falsey
        return True

    def __hash__(self) -> int:
        return id(self)

    def update_connection(self,
                          new: str, current: str | None = None,
                          auto_update: bool = True,
                          validate: bool = True,
                          ignore_case: bool = False,
        ) -> None:
        """Update the connection properties of the Layer.

        Args:
            new: The new connection string.
            current: The connection string to replace.
            auto_update: Update joins and relates.
            validate: Validate the Layer connection before setting it.
            ignore_case: Ignore case in queries.
        """
        self.elem.updateConnectionProperties(current, new, auto_update, validate, ignore_case)

    def select(self,
               *,
               predicate: Callable[[dict[str, Any]], bool] | None = None,
               query: str | None = None,
               method: mpt.Method | Literal['CLEAR', 'ALL'] = 'NEW',
    ) -> Self:
        """Set a selection on the Layer using a predicate or query and a method.

        Args:
            predicate: A function that takes a feature record and returns True if it is to be selected.
            query: A SQL where clause that will define the selection (ignored if `predicate` is set).
            method: How to resolve the selection with existing selection.

        Note:
            Calling with `method = 'CLEAR'` and no other arguments will clear the selection.
            Calling with `method = 'ALL'` and no other arguments will select all features.
        """
        if method == 'CLEAR':
            self.elem.setSelectionSet([])
            return self

        if method == 'ALL':
            self.elem.setSelectionSet(list(self.feature_class['OID@']))
            return self

        if predicate is not None:
            self.elem.setSelectionSet(
                [r['OID@'] for r in self.feature_class if predicate(r)],
                method
            )
            return self

        with self.query_as(query or self.definition_query):
            self.elem.setSelectionSet(list(self.feature_class['OID@']), method)

        return self

    def selected(self) -> list[dict[str, Any]]:
        """Get a list of all selected feature records."""
        oids = self.elem.getSelectionSet() or set()
        return [row for row in self.feature_class if row['OID@'] in oids]

    def unselected(self) -> list[dict[str, Any]]:
        """Get a list of all unselected feature records."""
        oids = self.elem.getSelectionSet() or set()
        return [row for row in self.feature_class if row['OID@'] not in oids]

    def clear_selection(self) -> Self:
        """Clear the current selection. (same as `layer.select(method='CLEAR'`)"""
        return self.select(method='CLEAR')

    def select_all(self) -> Self:
        """Select all features in the Layer. (same as `layer.select(method='ALL'`)"""
        return self.select(method='ALL')

    def export_lyrx(self, outdir: Path | str, *, name: str | None = None, indent: int | None = None, relative: bool = True) -> Path:
        """Export the `lyrx` definition of the Layer to a file.

        Args:
            outdir: The directory to place the `lyrx` file in.
            name: An optional name for the `lyrx` file. (default is `layer.short_name`)
            indent: Indentation level in the output `lyrx` file. (default: `None`)
            relative: Convert all absolute paths in the `lyrx` file to relative paths. (default: `True`)
        """
        outdir = Path(outdir)
        name = name or self.short_name
        outdir = outdir / name
        outdir = outdir.with_suffix('.lyrx')
        outdir.parent.mkdir(exist_ok=True, parents=True)
        outdir.write_text(json.dumps(self.relative_lyrx if relative else self.lyrx, indent=indent))
        return outdir

    def export_csv(self, outdir: Path | str,
                   *,
                   name: str | None = None, fields: Iterable[str] | None = None, sep: str = ',', newline: str = '\n') -> Path:
        """Export the table data to a CSV file.

        Args:
            outdir: The output directory or file.
            name: An optional name to give the file (default: `layer.name`)
        """
        outdir = Path(outdir)
        outfile = outdir.with_name(name or self.name).with_suffix('.csv')
        outfile.parent.mkdir(exist_ok=True, parents=True)
        table = self.feature_class
        fields = fields or table.fields
        with outfile.open('wt', encoding='utf-8', newline=newline) as fl:
            fl.write(sep.join(fields))
            fl.write(newline)
            fl.writelines(sep.join(line) for line in table[tuple(fields)])
        return outfile

    @property
    def label_classes(self) -> ElementList[LabelClass]:
        """Get an ElementList of all LabelClasses for the Layer."""
        return self._cached('label_classes',
            lambda: ElementList(
                LabelClass(lc, self)
                for lc in
                (
                    self.elem.listLabelClasses()
                    if 'showLabels' in self.supported
                    else []
                )
        ))

    @overload
    @warnings.deprecated('VBSCRIPT is deprecated, use ARCADE or PYTHON instead')
    def create_label_class(self, name: str, expression: str, query: str | None = ..., language: Literal['VBSCRIPT'] = 'VBSCRIPT') -> LabelClass: ...
    @overload
    def create_label_class(self, name: str, expression: str, query: str | None = ..., language: mpt.LabelClassLanguage = ...) -> LabelClass: ...
    def create_label_class(self, name: str, expression: str, query: str | None = None, language: mpt.LabelClassLanguage = 'ARCADE') -> LabelClass:
        """Create a new LabelClass for the Layer.

        Args:
            name: The name of the new LabelClass.
            expression: An expression to use for the LabelClass.
            query: An optional SQL where clause to filter the labeled features with.
            language: The expression language. (default: `ARCADE`)
        """
        if 'showLabels' in self.supported:
            lc = LabelClass(self.elem.createLabelClass(name, expression, query, language), self)
            self.refresh('label_classes')
            return lc
        return type(self).NotSupported  # type: ignore

    @property
    def charts(self) -> tuple[Chart, ...]:
        """Get Chart objects associated with the Layer."""
        try:
            return self._cached('charts',
                lambda: tuple(self.elem.listCharts())
            )
        except AttributeError:
            return ()

    @property
    def chart_map(self) -> ChartMap:
        """Get a mapping of all charts by type. (all chart types included, empty list for unused chart types)"""
        return cast(ChartMap, {
            chart_type: [c for c in self.charts if type(c).__name__ == chart_type]
            for chart_type in ChartTypes
        })

    def clone_properties(self, from_layer: LayerLike, properties: mpt.LayerPasteProperties = 'ALL'):
        """Copy specified properties from another Layer."""
        from_layer = from_layer.elem if isinstance(from_layer, Element) else from_layer
        self.elem.pasteProperties(from_layer, properties)

    def update_from_json(self, data: str | dict[str, Any]) -> None:
        """Update the Layer using a WebBuilder json definition."""
        if isinstance(data, dict):
            data = json.dumps(data)
        self.elem.updateLayerFromJSON(data)

    def open_view(self, selected: bool = False) -> None:
        """Open the Layer table view in the active project."""
        self.elem.openTableView(selected)


class LabelClass(Element[mpt.LabelClass, cim.CIMLabelClass, Layer]):
    lang_codes: ClassVar[dict[str, mpt.LabelClassLanguage]] = {
            'VBScript': 'VBSCRIPT',
            'JScript': 'JSCRIPT',
            'Python': 'PYTHON',
            'Arcade': 'ARCADE',
        }

    @property
    def expression(self) -> str:
        """A string representation of the label expression in the selected language."""
        return self.elem.expression

    @expression.setter
    def expression(self, expression: str) -> None:
        self.elem.expression = expression

    @property
    def query(self) -> str:
        """A SQL where clause to filter labeled features."""
        return self.elem.SQLQuery

    @query.setter
    def query(self, query: str | None) -> None:
        self.elem.SQLQuery = query or ''

    @property
    def visible(self) -> bool:
        """Visibility state of the LabelClass."""
        return self.elem.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self.elem.visible = visible

    @property
    def language(self) -> mpt.LabelClassLanguage:
        """The language code for the LabelClass"""
        lang: str = cast(str, self.cim.expressionEngine)
        return type(self).lang_codes.get(lang, 'ARCADE')

    @language.setter
    def language(self, language: mpt.LabelClassLanguage):
        if language == 'VBSCRIPT':
            warnings.warn(
                'VBSCRIPT is deprecated, use ARCADE or PYTHON instead',
                category=DeprecationWarning,
                stacklevel=2,
            )
        definition = self.cim
        definition.expressionEngine = cast(cim.LabelExpressionEngine, language)
        self.cim = definition


class Table(Element[mpt.Table, cim.CIMFeatureTable, Map | GroupLayer]):

    @property
    def source_data(self) -> fc.Table:
        """Alias for `table` that allows using Tables and Layers in duck typed loops."""
        return self.table

    @property
    def table(self) -> fc.Table:
        """The base Table object for the Table Layer."""
        return fc.Table.from_layer(cast(mpt.Layer, self.elem))

    @property
    def cim_dict(self) -> dict[str, Any]:
        """CIM definition of the Table in dictionary format."""
        cim_dict = super().cim_dict
        # Convert absolute paths to relative databases into relative paths
        # This allows sharing layers bw projects with the same structure
        if (
            (ft := cim_dict.get('featureTable'))
            and (conn := ft.get('dataConnection'))
            and (ws_conn := conn.get('workspaceConnectionString'))
            and (tbl := self.table)
        ):
            cur_path = Path(ws_conn.repace('DATABASE=', ''))
            database = Path(tbl.workspace)
            if cur_path.is_absolute() and cur_path.is_relative_to(database.parent):
                conn['workspaceConnectionString'] = f'DATABASE={cur_path.relative_to(database.parent)}'
        return cim_dict

    @property
    def lyrx(self) -> dict[str, Any]:
        """A dictionary representation of the table `lyrx` file."""
        lyrx = dict[str, Any]()
        lyrx['type'] = 'CIMLayerDocument'
        lyrx['tables'] = [self.uri]
        lyrx['layerDefinitions'] = [self.cim_dict]
        return lyrx

    @property
    def definition_queries(self) -> list[DefinitionQuery]:
        """A list of all DefinitionQueries for the Table."""
        return cast(list[DefinitionQuery], self.elem.listDefinitionQueries())

    @definition_queries.setter
    def definition_queries(self, queries: Iterable[DefinitionQuery]) -> None:
        queries = list(queries)
        if total_active := sum(1 for q in queries if q.get('isActive')) > 1:
            raise ValueError(f'{total_active} queries are set to active (max: 1)')
        self.elem.updateDefinitionQueries(cast(list[dict[str, str | bool]], queries))

    @property
    def definition_query(self):
        """The SQL string of the active DefinitionQuery."""
        return self.elem.definitionQuery

    @definition_query.setter
    def definition_query(self, query: str | None) -> None:
        self.elem.definitionQuery = query or ''

    @property
    def charts(self) -> tuple[Chart, ...]:
        """All Chart objects associated with the Table."""
        return self._cached('charts',
            lambda: tuple(self.elem.listCharts())
        )

    @property
    def chart_map(self) -> ChartMap:
        """Chart objects for the Table grouped by chart type. (all types included, empty list for types that don't exist)"""
        return cast(ChartMap, {
            chart_type: [c for c in self.charts if type(c).__name__ == chart_type]
            for chart_type in ChartTypes
        })

    @property
    def data_source(self) -> str:
        """The Table data source as a string."""
        return self.elem.dataSource

    @data_source.setter
    def data_source(self, source: str | fc.FeatureClass) -> None:
        if isinstance(source, fc.FeatureClass):
            source = source.path
        self.update_connection(source)

    @property
    def selection(self) -> set[int]:
        """Get the selected record OID set."""
        return set(self.table['OID@'])

    @selection.setter
    def selection(self, selection: Iterable[int]) -> None:
        self.elem.setSelectionSet(list(set(selection)))

    @contextmanager
    def query_as(self, query: str | None):
        """Context manager for setting the active DefinitionQuery string.

        Example:
        ```python
        >>> print(len(table))
        250
        >>> with lay.query_as("STATUS = 'ACTIVE'"):
        ...     print(len(table))
        125
        ```
        """
        cur = self.definition_query
        try:
            self.definition_query = query
            yield self
        finally:
            self.definition_query = cur

    def __len__(self) -> int:
        return len(self.table)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.table)

    def __bool__(self) -> Literal[True]:
        # Override __bool__ to prevent __iter__/__len__ from making a table falsey
        return True

    def update_connection(self,
                          new: str,
                          current: str | None = None,
                          auto_update: bool = True,
                          validate: bool = True,
                          ignore_case: bool = False,
        ) -> None:
        """Update the connection properties of the Table.

        Args:
            new: The new connection string.
            current: The connection string to replace.
            auto_update: Update joins and relates.
            validate: Validate the Table connection before setting it.
            ignore_case: Ignore case in queries.
        """
        self.elem.updateConnectionProperties(current, new, auto_update, validate, ignore_case)

    def select(self,
               *,
               predicate: Callable[[dict[str, Any]], bool] | None = None,
               query: str | None = None,
               method: mpt.Method | Literal['CLEAR', 'ALL'] = 'NEW',
    ) -> Self:
        """Set a selection on the Table using a predicate or query and a method.

        Args:
            predicate: A function that takes a feature record and returns True if it is to be selected.
            query: A SQL where clause that will define the selection (ignored if `predicate` is set).
            method: How to resolve the selection with existing selection.

        Note:
            Calling with `method = 'CLEAR'` and no other arguments will clear the selection.
            Calling with `method = 'ALL'` and no other arguments will select all features.
        """
        if method == 'CLEAR':
            self.elem.setSelectionSet([])
            return self

        if method == 'ALL':
            self.elem.setSelectionSet(list(self.table['OID@']))
            return self

        if predicate is not None:
            self.elem.setSelectionSet(
                [r['OID@'] for r in self.table if predicate(r)],
                method
            )
            return self

        with self.query_as(query or self.definition_query):
            self.elem.setSelectionSet(list(self.table['OID@']), method)
        return self

    def selected(self) -> list[dict[str, Any]]:
        """Get a list of all selected table records."""
        oids = self.elem.getSelectionSet() or set()
        return [row for row in self.table if row['OID@'] in oids]

    def unselected(self) -> list[dict[str, Any]]:
        """Get a list of all unselected table records."""
        oids = self.elem.getSelectionSet() or set()
        return [row for row in self.table if row['OID@'] not in oids]

    def clear_selection(self) -> Self:
        """Clear the current selection."""
        return self.select(method='CLEAR')

    def select_all(self) -> Self:
        """Select all rows in the Table."""
        return self.select(method='ALL')

    def export_lyrx(self,
                    outdir: Path | str,
                    *,
                    name: str | None = None,
                    indent: int | None = None,
        ) -> Path:
        """Export the `lyrx` definition of the Table to a file.

        Args:
            outdir: The directory to place the `lyrx` file in.
            name: An optional name for the `lyrx` file. (default is `layer.short_name`)
            indent: Indentation level in the output `lyrx` file. (default: `None`)
            relative: Convert all absolute paths in the `lyrx` file to relative paths. (default: `True`)
        """
        outdir = Path(outdir)
        name = name or self.short_name
        outdir = outdir / name
        outdir = outdir.with_suffix('.lyrx')
        outdir.parent.mkdir(exist_ok=True, parents=True)
        outdir.write_text(json.dumps(self.lyrx, indent=indent))
        return outdir

    def export_csv(self, outdir: Path | str,
                   *,
                   name: str | None = None,
                   fields: Iterable[str] | None = None,
                   sep: str = ',',
                   newline: str = '\n',
        ) -> Path:
        """Export the table data to a CSV file.

        Args:
            outdir: The output directory or file.
            name: An optional name to give the file (default: `table.name`)
        """
        outdir = Path(outdir)
        outfile = outdir.with_name(name or self.name).with_suffix('.csv')
        outfile.parent.mkdir(exist_ok=True, parents=True)
        table = self.table
        fields = fields or table.fields
        with outfile.open('wt', encoding='utf-8', newline=newline) as fl:
            fl.write(sep.join(fields))
            fl.write(newline)
            fl.writelines(sep.join(line) for line in table[tuple(fields)])
        return outfile

    def clone_properties(self, from_table: TableLike, properties: mpt.TablePasteProperties = 'ALL'):
        """Copy specified properties from another Table."""
        from_table = from_table.elem if isinstance(from_table, Element) else from_table
        self.elem.pasteProperties(from_table, properties)

    def open_view(self, show_selected: bool = False) -> None:
        """Open the Table view in the avtive project."""
        self.elem.openTableView(show_selected)

    def delete(self) -> None:
        """Delete the table from the Map."""
        parent = self.parent
        while parent and not isinstance(parent, Map):
            parent = parent.parent
        if parent is None:
            raise ValueError(f'{self} has no associated Map')
        else:
            parent.remove(self)


class ElevationSurface(Element[mpt.ElevationSurface, cim.CIMLayerElevationSurface, Map]):

    @property
    def map(self) -> Map:
        """The Map that the ElevationSurface is associated with."""
        if not self.parent:
            raise AttributeError(f'{self} has no associated Map')
        return self.parent

    @property
    def has_map(self) -> bool:
        """Determine if the ElevationSurface is associated with a Map."""
        return self.parent is not None

    @property
    def vertical_exaggeration(self) -> float:
        """Vertical exaggeration of the Elevation Z values."""
        return self.elem.verticalExaggeration

    @vertical_exaggeration.setter
    def vertical_exaggeration(self, ve: float) -> None:
        self.elem.verticalExaggeration = ve

    @property
    def elevation_sources(self) -> ElementList[ElevationSource]:
        """ElementList of all ElevationSources for the ElevationSurface."""
        return self._cached('elevation_sources',
            lambda: ElementList(
                ElevationSource(es, self.map.parent if self.has_map else None)
                for es in self.elem.listElevationSources()
        ))


class Layout(Element[mpt.Layout, cim.CIMLayout, Project]):

    @property
    def elements(self) -> ElementList[LayoutElement[Any, Any]]:
        """ElementList of all LayoutElements in the Layout. (use other accessors for typed/filtered Elements.)"""
        return self._cached('elements',
            lambda: ElementList(
                LayoutElement[Any, Any](e, self)
                for e in self.elem.listElements()
            )
        )

    @property
    def graphic_elements(self) -> ElementList[GraphicElement]:
        """ElementList of all GraphicElements in the Layout."""
        return ElementList(
            GraphicElement(cast(mpt.GraphicElement, e.elem), self)
            for e in self.elements
            if e.mp_type_name == 'GraphicElement'
        )

    @property
    def group_elements(self) -> ElementList[GroupElement]:
        """ElementList of all GroupElements in the Layout."""
        return ElementList(
            GroupElement(cast(mpt.GroupElement, e.elem), self)
            for e in self.elements
            if e.mp_type_name == 'GroupElement'
        )

    @property
    def map_frames(self) -> ElementList[MapFrame]:
        """ElementList of all MapFrames in the Layout."""
        return ElementList(
            MapFrame(cast(mpt.MapFrame, e.elem), self)
            for e in self.elements
            if e.mp_type_name == 'MapFrame'
        )

    @property
    def map_surround_elements(self) -> ElementList[MapSurroundElement]:
        """ElementList of all MapSurroundElements in the Layout."""
        return ElementList(
            MapSurroundElement(cast(mpt.MapSurroundElement, e.elem), self)
            for e in self.elements
            if e.mp_type_name == 'MapSurroundElement'
        )

    @property
    def picture_elements(self) -> ElementList[PictureElement]:
        """ElementList of all PictureElements in the Layout."""
        return ElementList(
            PictureElement(cast(mpt.PictureElement, e.elem), self)
            for e in self.elements
            if e.mp_type_name == 'PictureElement'
        )

    @property
    def table_frame_elements(self) -> ElementList[TableFrameElement]:
        """ElementList of all TableFrameElements in the Layout."""
        return ElementList(
            TableFrameElement(cast(mpt.TableFrameElement, e.elem), self)
            for e in self.elements
            if e.mp_type_name == 'TableFrameElement'
        )

    @property
    def text_elements(self) -> ElementList[TextElement]:
        """ElementList of all TextElements in the Layout."""
        return ElementList(
            TextElement(cast(mpt.TextElement, e), self)
            for e in self.elements
            if e.mp_type_name == 'TextElement'
        )

    @property
    def mapseries(self) -> MapSeries:
        """The spatial Mapseries for the Layout if it exists. (raises: `AttributeError`)"""
        if not self.has_mapseries:
            raise AttributeError(f'{self} has no associated MapSeries')
        return MapSeries(cast(mpt.MapSeries, self.elem.mapSeries), self)

    @property
    def has_mapseries(self) -> bool:
        """Determine if the Layout has a spatial Mapseries."""
        return type(self.elem.mapSeries).__name__ == 'MapSeries'

    @property
    def bookmark_mapseries(self) -> BookmarkMapSeries:
        """The BookmarkMapseries for the Layout if it exists. (raises: `AttributeError`)"""
        if not self.has_bookmark_mapseries:
            raise AttributeError(f'{self} has no associated BookmarkMapSeries')
        return BookmarkMapSeries(cast(mpt.BookmarkMapSeries, self.elem.mapSeries), self)

    @property
    def has_bookmark_mapseries(self) -> bool:
        """Determine if the Layout has a BookmarkMapseries."""
        return type(self.elem.mapSeries).__name__ == 'BookmarkMapSeries'

    @property
    def width(self) -> float:
        """The Layout width in page units."""
        return self.elem.pageWidth

    @width.setter
    def width(self, width: float) -> None:
        self.elem.pageWidth = width

    @property
    def height(self) -> float:
        """The Layout height in page units."""
        return self.elem.pageHeight

    @height.setter
    def height(self, height: float) -> None:
        self.elem.pageHeight = height

    @property
    def units(self) -> mpt.PageUnits:
        """The Layout page units."""
        return self.elem.pageUnits

    @units.setter
    def units(self, units: mpt.PageUnits) -> None:
        self.elem.pageUnits = units

    @property
    def color_model(self) -> mpt.ColorModel:
        """The Layout width in ColorModel."""
        return self.elem.colorModel

    @color_model.setter
    def color_model(self, color_model: mpt.ColorModel) -> None:
        self.elem.colorModel = color_model

    @property
    def metadata(self) -> Metadata:
        """The Layout Metadata."""
        return self.elem.metadata

    @metadata.setter
    def metadata(self, metadata: Metadata) -> None:
        self.elem.metadata = metadata

    def create_mapseries(
        self,
        frame: MapFrameLike,
        layer: LayerLike,
        name_field: str,
        sort_field: str | None = None,
    ) -> MapSeries:
        """Create a spatial Mapseries for the Layout.

        Args:
            frame: The MapFrame to use for the Mapseries.
            layer: The series layer in the MapFrame Map.
            name_field: The field to use as the Mapseries page name.
            sort_field: The field to sorth the Maperies pages by (default: `name_field`).
        """
        if isinstance(frame, MapFrame):
            frame = frame.elem
        if isinstance(layer, Layer):
            layer = layer.elem
        return MapSeries(self.elem.createSpatialMapSeries(frame, layer, name_field, sort_field), self)

    def create_bookmark_mapseries(
        self,
        frame: MapFrameLike,
        bookmarks: Iterable[Bookmark] | Iterable[mpt.Bookmark] | None = None,
    ) -> BookmarkMapSeries:
        """Create a BookmarkMapseries for the Layout.

        Args:
            frame: The MapFrame to use for the BookmarkMapseries (must have Bookmarks).
            bookmarks: The Bookmarks in the Map to use. (default: All)
        """
        if isinstance(frame, MapFrame):
            frame = frame.elem
        bookmarks = [
            b.elem if isinstance(b, Bookmark) else b
            for b in bookmarks or []
        ]
        return BookmarkMapSeries(self.elem.createBookmarkMapSeries(frame, bookmarks), self)

    def create_map_frame(
        self,
        *,
        name: str | None = None,
        map: MapLike | None = None,
        geometry: Polygon | Point | HasCentroid | None = None,
    ) -> MapFrame:
        """Create a new MapFrame in the Layout.

        Args:
            geometry: An anchor point or polygon shape for the MapFrame (using page units). (default: `Point(0,0)`)
            map: The Map to associate the MapFrame with.
            name: The name of the MapFrame element.

        Notes:
            Use a Polygon created in Layout coordinates to specify an exact shape, or use a shape with a centroid
            to set the anchor point of the MapFrame.
            After creating the MapFrame, the anchor point, width, and height can be set directly.
            If no geometry is used, the MapFrame will be created at the Layout origin.
        """
        if geometry is None:
            geometry = Point(0, 0)
        if not isinstance(geometry, Polygon | Point):
            if not isinstance(cast(Any, geometry), HasCentroid):
                raise ValueError(
                    f'Invalid source geometry {type(geometry).__name__} for MapFrame, '
                    'must have trueCentroid, centroid and spatialReference attributes'
                )
            geometry = geometry.centroid
        if isinstance(map, Map):
            map = map.elem
        return MapFrame(self.elem.createMapFrame(geometry, map, name), self)

    def create_map_surround_element(
        self,
        surround_type: mpt.MapSurroundType,
        *,
        name: str | None = None,
        frame: MapFrameLike | None = None,
        geometry: Polygon | Point | HasCentroid | None = None,
        style_item: StyleItemLike | None = None,
    ) -> MapSurroundElement:
        """Create a new MapSurroundElement in the Layout.

        Args:
            name: An optional name for the MapSurroundElement.
            surround_type: The MapSurroundElement type.
            geometry: An anchor point or polygon shape for the MapSurroundElement (using page units). (default: `Point(0,0)`)
            frame: The MapFrame to associate the MapSurroundElement with.
            style_item: A style item to use for the MapSurroundElement style.

        Notes:
            Use a Polygon created in Layout coordinates to specify an exact shape, or use a shape with a centroid
            to set the anchor point of the MapSurroundElement.
            After creating the MapSurroundElement, the anchor point, width, and height can be set directly.
            If no geometry is used, the MapSurroundElement will be created at the Layout origin.
        """
        if geometry is None:
            geometry = Point(0, 0)
        if not isinstance(geometry, Polygon | Point):
            if not isinstance(cast(Any, geometry), HasCentroid):
                raise ValueError(
                    f'Invalid source geometry {type(geometry).__name__} for MapFrame, '
                    'must have trueCentroid, centroid and spatialReference attributes'
                )
            geometry = geometry.centroid
        if isinstance(frame, MapFrame):
            frame = frame.elem
        if isinstance(style_item, StyleItem):
            style_item = style_item.elem
        return MapSurroundElement(self.elem.createMapSurroundElement(geometry, surround_type, frame, style_item, name), self)

    def create_table_frame_element(
        self,
        *,
        name: str | None = None,
        frame: MapFrameLike | None = None,
        table: LayerLike | TableLike | None = None,
        fields: Iterable[str] | None = None,
        geometry: Point | Polygon | HasCentroid | None = None,
        style_item: StyleItemLike | None = None,
    ) -> TableFrameElement:
        """Create a new TableFrameElement in the Layout.

        Args:
            name: An optional name for the TableFrameElement.
            geometry: An anchor point or polygon shape for the TableFrameElement (using page units). (default: `Point(0,0)`)
            frame: The MapFrame to associate the TableFrameElement with.
            table: The Table or Layer in the MapFrame to use as the TableFrameElement source.
            fields: A list of fields to include in the TableFrameElement.
            style_item: A style item to use for the TableFrameElement style.

        Notes:
            Use a Polygon created in Layout coordinates to specify an exact shape, or use a shape with a centroid
            to set the anchor point of the TableFrameElement.
            After creating the TableFrameElement, the anchor point, width, and height can be set directly.
            If no geometry is used, the TableFrameElement will be created at the Layout origin.
        """
        if geometry is None:
            geometry = Point(0, 0)
        if not isinstance(geometry, Polygon | Point):
            if not isinstance(cast(Any, geometry), HasCentroid):
                raise ValueError(
                    f'Invalid source geometry {type(geometry).__name__} for MapFrame, '
                    'must have trueCentroid, centroid and spatialReference attributes'
                )
            geometry = geometry.centroid
        if isinstance(frame, MapFrame):
            frame = frame.elem
        if isinstance(style_item, StyleItem):
            style_item = style_item.elem
        if isinstance(table, Layer | Table):
            table = table.elem
        fields = list(fields) if fields else None
        return TableFrameElement(self.elem.createTableFrameElement(geometry, frame, table, fields, style_item, name), self)

    def delete_element(self, elem: LayoutElement[mpt.LayoutElement, Any] | mpt.LayoutElement):
        """Delete an element from the Layout."""
        if isinstance(elem, LayoutElement):
            elem = elem.elem
        self.elem.deleteElement(elem)

    def delete_elements(self, *elems: LayoutElement[mpt.LayoutElement, Any] | mpt.LayoutElement) -> None:
        """Delete multiple elements from the Layout (varargs)."""
        for elem in elems:
            self.delete_element(elem)

    def resize(self, width: float | None = None, height: float | None = None, resize_elements: bool = True) -> None:
        """Resize the Layout.

        Args:
            width: The new Layout width.
            height: The new Layout height.
            resize_elements: Resize all child elements to maintain scale. (default: `True`)
        """
        width = width or self.width
        height = height or self.height
        self.elem.changePageSize(width, height, resize_elements)

    @overload
    def export(self,  # type: ignore (No Overlap?)
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: None = None,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> bytes: ...
    @overload
    def export(self,
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: Path | str = ...,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> Path: ...
    def export(self,
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: Path | str | None = None,
               antialiasing: mpt.Antialiasing | None = None,
        ) -> Path | bytes:
        """Export the Layout.

        Args:
            format: A Format object or a string format (string option will use defaults)
            outfile: An optional output file location to use (will override format filePath)
            antialiasing: Antialiasing options for the output file

        Returns:
            Path: If outfile is set
            bytes: If outfile is unset
        """
        display = None
        outfile = Path(outfile) if outfile else None
        if antialiasing:
            display = cast(mpt.DisplayOptions, mp.CreateExportOptions('DISPLAY'))
            display.setAntialiasing(antialiasing)

        if isinstance(format, fmts.Format):
            format = format.fmt

        if type(format).__name__.endswith('Format'):
            suffix = type(format).__name__.replace('Format', '').lower()
            format = cast(mpt.ExportFormat, format)

        elif isinstance(format, str):
            suffix = format.lower()
            format = mp.CreateExportFormat(format)

        else:
            raise ValueError(f'Unknown format {type(format)} : {format}')

        if suffix == 'jpeg':
            suffix = 'jpg'

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / f'{self.name}.{suffix}'
            format.filePath = str(out)
            self.elem.export(format, display_options=display)
            data = out.read_bytes()
        if outfile:
            outfile.parent.mkdir(exist_ok=True, parents=True)
            outfile.with_suffix(suffix).write_bytes(data)
            return outfile
        else:
            return data

    def open_view(self):
        """Open the Layout in the active Project."""
        self.elem.openView()


class MapFrame(Element[mpt.MapFrame, cim.CIMMapFrame, Layout]):

    @property
    def map(self) -> Map:
        """The Map associated with the MapFrame."""
        map = self.elem.map
        layout = self.parent
        if map and layout:
            return Map(map, layout.parent)
        raise ValueError(f'{self} has no associated Map')

    @property
    def parent_group(self) -> GroupElement:
        """The GroupElement the MapFrame is a child of if it exists. (raises: `AttributeError`)"""
        if pg := self.elem.parentGroupElement:
            return GroupElement(pg, self.parent)
        raise AttributeError(f'{self} is not part of a GroupElement')

    @property
    def in_group(self) -> bool:
        """Determine if the MapFrame is part of a GroupElement."""
        return self.elem.parentGroupElement is not None

    @property
    def has_map(self) -> bool:
        """Check if the MapFrame has an associated Map"""
        return self.elem.map is not None

    @property
    def alt_text(self) -> str:
        """The hover text for the MapFrame."""
        return self.elem.altText

    @alt_text.setter
    def alt_text(self, text: str) -> None:
        self.elem.altText = text

    @property
    def anchor(self) -> mpt.Anchor:
        """The anchor point of the MapFrame."""
        return self.elem.anchor

    @anchor.setter
    def anchor(self, anchor: mpt.Anchor) -> None:
        self.elem.anchor = anchor

    @property
    def camera(self) -> mp.Camera:
        """The current camera of the MapFrame."""
        return self.elem.camera

    @camera.setter
    def camera(self, camera: mp.Camera) -> None:
        self.elem.camera = camera

    @property
    def visible(self) -> bool:
        """Visibility state of the MapFrame in the Layout."""
        return self.elem.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self.elem.visible = visible

    @property
    def locked(self) -> bool:
        """Editability status of the MapFrame in the Layout."""
        return self.elem.locked

    @locked.setter
    def locked(self, locked: bool) -> None:
        self.elem.locked = locked

    @property
    def x(self) -> float:
        return self.elem.elementPositionX

    @x.setter
    def x(self, x: float) -> None:
        """The X coordinate of the anchor point in page units."""
        self.elem.elementPositionX = x

    @property
    def y(self) -> float:
        """The Y coordinate of the anchor point in page units."""
        return self.elem.elementPositionY

    @y.setter
    def y(self, y: float) -> None:
        self.elem.elementPositionY = y

    @property
    def width(self) -> float:
        """The width of the MapFrame in page units."""
        return self.elem.elementWidth

    @width.setter
    def width(self, width: float) -> None:
        self.elem.elementWidth = width

    @property
    def height(self) -> float:
        """The height of the MapFrame in page units."""
        return self.elem.elementHeight

    @height.setter
    def height(self, height: float) -> None:
        self.elem.elementHeight = height

    @property
    def rotation(self) -> float:
        """The rotation of the MapFrame in degrees (use `camera.heading` for view rotation)."""
        return self.elem.elementRotation

    @rotation.setter
    def rotation(self, rotation: float) -> None:
        self.elem.elementRotation = rotation

    def add_grid(self, style_item: StyleItemLike):
        """Add a grid or graticule to the MapFrame.

        Args:
            style_item: The style item to create the grid with (must be a `GRID` class style).
        """
        style_item = style_item.elem if isinstance(style_item, Element) else style_item
        self.elem.addGrid(style_item)

    def remove_grids(self, grid: str | None = None) -> None:
        """Remove grids from the MapFrame.

        Args:
            grid: A wildcard string that uses the Grid name in the contents pane.
        """
        self.elem.removeGrids(grid)

    def convert_grid(self, grid: str, out_gdb: Path | str, new_name: str | None = None) -> None:
        """Convert a MapFrame grid into real geometry in the associated Map.

        Args:
            grid: The name of the grid to export.
            out_gdb: The geodatabase to export the grid features to.
            new_name: The name of the group layer that the features will be added to (default: grid name).
        """
        self.elem.convertGridToFeatures(grid, str(out_gdb), new_name)

    def create_bookmark(self, name: str | None = None, description: str | None = None) -> Bookmark:
        return Bookmark(self.elem.createBookmark(name, description), self.map)

    @overload
    def export(self,  # type: ignore (No Overlap?)
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: None = None,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> bytes: ...
    @overload
    def export(self,
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: Path | str = ...,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> Path: ...
    def export(self,
               format: mp.Format | mpt.ExportFormat | fmts.Format,
               *,
               outfile: Path | str | None = None,
               antialiasing: mpt.Antialiasing | None = None,
        ) -> Path | bytes:
        """Export the MapFrame. (wrapper for `MapView.export`)

        Args:
            format: A Format object or a string format (string option will use defaults)
            outfile: An optional output file location to use (will override format filePath)
            antialiasing: Antialiasing options for the output file
        """
        display = None
        outfile = Path(outfile) if outfile else None
        if antialiasing:
            display = cast(mpt.DisplayOptions, mp.CreateExportOptions('DISPLAY'))
            display.setAntialiasing(antialiasing)

        if isinstance(format, fmts.Format):
            format = format.fmt

        if type(format).__name__.endswith('Format'):
            suffix = type(format).__name__[:3].lower()
            format = cast(mpt.ExportFormat, format)

        elif isinstance(format, str):
            suffix = format[:3].lower()
            format = mp.CreateExportFormat(format)

        else:
            raise ValueError(f'Unknown format {type(format)} : {format}')

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / f'{self.name}.{suffix}'
            format.filePath = str(out)
            self.elem.export(format, display_options=display)
            data = out.read_bytes()
        if outfile:
            outfile.parent.mkdir(exist_ok=True, parents=True)
            outfile.with_suffix(suffix).write_bytes(data)
            return outfile
        else:
            return data

    def layer_extent(self, layer: LayerLike, selected: bool = True, symbolized: bool = True) -> Extent:
        """Get the extent of a layer within the MapFrame view.

        Args:
            layer: The layer to get an extent for (must be in MapFrame map).
            selected: Get the extent of selected features only.
            symbolized: Get the extent of symbology at current scale, not real shape.
        """
        layer = layer.elem if isinstance(layer, Element) else layer
        return self.elem.getLayerExtent(layer, selected, symbolized)

    def pan_to(self, extent: Extent | LayerLike | HasExtent) -> None:
        """Pan the MapFrame view to an object or extent."""
        if type(extent).__name__ == 'Layer':
            extent = self.layer_extent(cast(mpt.Layer, extent))
        if isinstance(extent, HasExtent):
            extent = extent.extent
        self.elem.panToExtent(cast(Extent, extent))

    def zoom_all(self, selected: bool = True, symbolized: bool = True) -> None:
        """Zoom the MapFrame to fit all Map layers.

        Args:
            selected: Zoom to the extent of the selected features only.
            symbolized: Zoom to the extent of the Layer symbologies.
        """
        self.elem.zoomToAllLayers(selected, symbolized)

    def zoom_to(self, elem: LayerLike | BookmarkLike | HasExtent) -> None:
        """Zoom the MapFrame to a Layer/Bookmark/Extent (or extent having object)."""
        if isinstance(elem, Element):
            elem = elem.elem

        elem_type = type(elem).__name__
        if elem_type == 'Bookmark':
            self.elem.zoomToBookmark(cast(mpt.Bookmark, elem))
        if elem_type == 'Layer':
            self.camera.setExtent(self.layer_extent(cast(mpt.Layer, elem)))
        else:
            self.camera.setExtent(cast(HasExtent, elem).extent)


# MapSeries export only allows a subset of ExportFormats
type _MSFormat = mpt.PDFFormat | mpt.PNGFormat | mpt.JPEGFormat | mpt.TIFFFormat
type _MSFormatAlt = fmts.PDF | fmts.PNG | fmts.JPEG | fmts.TIFF
_MSFormatName = Literal['PDF', 'PNG', 'JPEG', 'TIFF']


class MapSeries(Element[mpt.MapSeries, cim.CIMMapSeries, Layout]):

    @property
    def enabled(self) -> bool:
        """Determine if the MapSeries is enabled."""
        return self.elem.enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self.elem.enabled = enabled

    @property
    def clip_to_features(self) -> bool:
        """Determine if the MapSeries will clip features to the MapFrame."""
        return self.elem.clipToIndexFeature

    @clip_to_features.setter
    def clip_to_features(self, clip_to_features: bool) -> None:
        self.elem.clipToIndexFeature = clip_to_features

    @property
    def map_frame(self) -> MapFrame:
        """The target MapFrame for the MapSeries."""
        return MapFrame(self.elem.mapFrame, self.parent)

    @property
    def layer(self) -> Layer:
        """The sereies Layer for the MapSeries"""
        return Layer(self.elem.indexLayer, self.map_frame.map)

    @property
    def page_name_field(self) -> str:
        """The name field for the MapSeries."""
        return self.elem.pageNameField.name

    @property
    def current_page_name(self) -> str:
        """The name of the current MapSeries page."""
        try:
            return str(self.elem.currentPageName)
        except Exception:
            # <=3.6 compat
            return str(self.page_row[self.page_name_field])

    @property
    def page_map(self) -> dict[str, str]:
        """A mapping of all page numbers to page names."""
        return dict(zip((str(i) for i in self.page_range), self.page_names, strict=False))

    @property
    def page_range(self) -> range:
        """A range object representing the page numbers (e.g. 10 pages == `range(1, 11)`)"""
        return range(1, self.page_count + 1)

    @property
    def current_page_number(self) -> int | str:
        """The number of the current MapSeries page."""
        return self.elem.currentPageNumber

    # TODO: Are these always going to be a closed range of integers?
    #       Or are they driven by the PageNumber property of the mapseries
    @property
    def page_numbers(self) -> list[str]:
        """String values for all page numbers in the MapSeries."""
        return [str(p.current_page_number) for p in self]

    @property
    def page_names(self) -> list[str]:
        """String values for all page names in the MapSeries."""
        return self._cached('page_names',
            lambda: [str(p.current_page_name) for p in self]
        )

    @property
    def name(self) -> str:
        """Name override for MapSeries, set the name with `current_page_name`"""
        return str(self.current_page_name)

    @name.setter  # name of a mapseries is determined by current page
    def name(self, name: Never) -> Never: ...  # type: ignore

    @property
    def page_count(self) -> int:
        """Total pages in the MapSeries."""
        return self.elem.pageCount

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        """A mapping of page numbers (strings) to feature records."""
        return self._cached('features',
            lambda: {
                page: layout.page_row
                for page, layout in zip(self.page_numbers, self, strict=True)
        })

    @property
    def page_row(self) -> dict[str, Any]:
        """Return a dictionary representing the current map page feature."""
        pr = cast(Any, self.elem.pageRow)
        if isinstance(pr, tuple):
            return pr._asdict()  # type: ignore (NamedTuple)
        if isinstance(pr, dict):
            return cast(dict[str, Any], pr)
        else:
            return pr

    @overload
    def export(self,  # type: ignore (No Overlap?)
               format: _MSFormat | _MSFormatAlt | _MSFormatName,
               *,
               out: None = None,
               antialiasing: mpt.Antialiasing | None = ...,
               mapseries_options: mpt.MapSeriesExportOptions | None = ...,
               custom_pages: str | None = ...,
               multi_file: bool = ...,
               export_pages: mpt.ExportPages = ...,
               print_count: bool = ...,
        ) -> tuple[bytes, ...]: ...
    @overload
    def export(self,
               format: _MSFormat | _MSFormatAlt | _MSFormatName,
               *,
               out: Path | str = ...,
               antialiasing: mpt.Antialiasing | None = ...,
               mapseries_options: mpt.MapSeriesExportOptions | None = ...,
               custom_pages: str | None = ...,
               multi_file: bool = ...,
               export_pages: mpt.ExportPages = ...,
               print_count: bool = ...,
        ) -> tuple[Path, ...]: ...
    def export(self,
               format: _MSFormat | _MSFormatAlt | _MSFormatName,
               *,
               out: Path | str | None = None,
               antialiasing: mpt.Antialiasing | None = None,
               mapseries_options: mpt.MapSeriesExportOptions | None = None,
               custom_pages: str | None = None,
               multi_file: bool = False,
               export_pages: mpt.ExportPages = 'ALL',
               prefix: str | None = None,
               print_count: bool = False,
        ) -> tuple[Path, ...] | tuple[bytes, ...]:
        """Export the MapSeries.

        Args:
            format: A string name for the output format or a Formatter object.
            out: An optional output file for the export.
            antialiasing: Antialiazing level for the export.
            mapseries_options: MapSeriesExportOptions object (additional kwargs will set this too.)
            custom_pages: An optional custom page range string for the export.
            multi_file: Export the mapseries as multiple files (one per page).
            export_pages: Page export type (All, Current, Selected, etc.)
            prefix: A prefix to add to the output files.
            print_count: When exporting with `multi_file` enabled, show export status with print/AddMessage.
        """
        ms_opts = mapseries_options or mp.CreateExportOptions('MAPSERIES')
        ms_opts = cast(mpt.MapSeriesExportOptions, ms_opts)
        ms_opts.showExportCount = print_count
        if export_pages == 'CUSTOM' or custom_pages:
            if not custom_pages:
                raise ValueError("`export_pages` is set to 'custom', but no `custom_pages` argument given")
            ms_opts.customPages = custom_pages
            ms_opts.setExportPages('CUSTOM')
        elif multi_file:
            ms_opts.setExportFileOptions('MULTIPLE_FILES_PAGE_NUMBER')
        elif export_pages == 'ALL':
            ms_opts.setExportPages('ALL')
        elif export_pages == 'CURRENT':
            ms_opts.setExportPages('CURRENT')
        elif export_pages == 'SELECTED_INDEX_FEATURES':
            ms_opts.setExportPages('SELECTED_INDEX_FEATURES')

        display = None
        out = Path(out) if out else None

        if antialiasing:
            display = cast(mpt.DisplayOptions, mp.CreateExportOptions('DISPLAY'))
            display.setAntialiasing(antialiasing)

        if isinstance(format, fmts.Format):
            format = format.fmt

        if type(format).__name__.endswith('Format'):
            suffix = type(format).__name__[:3].lower()
            format = cast(_MSFormat, format)

        elif isinstance(format, str):
            if format not in _MSFormatName.__args__:
                raise ValueError(f'{self} can only export to {_MSFormatName.__args__}, not {format}')
            # Some type forcing required here for the subset
            suffix = format[:3].lower()
            format = cast(_MSFormat, mp.CreateExportFormat(format))

        else:
            raise ValueError(f'Unknown format {type(format)} : {format}')

        out_name = f'MSPage.{suffix}' if multi_file else f'{self.name}.{suffix}'
        page_names = self.page_names if multi_file else []
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            outfl = tmp / out_name
            format.filePath = str(outfl)
            self.elem.export(format, mapseries_export_options=ms_opts, display_options=display)
            pages = tuple(
                (name, fl.read_bytes())
                for name, fl in zip(
                    page_names,
                    sorted(tmp.glob(f'*.{suffix}'), key=lambda f: f.stat().st_birthtime_ns),
                    strict=True
                )
            ) if multi_file else ((out_name, outfl.read_bytes()),)

        # Return the raw page data if an outfile is not specified
        if out is None:
            return tuple(page[1] for page in pages)

        paths = list[Path]()

        # If out is a pdf and there is only one page, write to it
        if len(pages) == 1:
            fl_name, data = pages[0]
            fl_name = f'{prefix or ''}{fl_name}'
            outfl = (
                out
                if out.suffix == f'.{suffix}'
                else (out / fl_name).with_suffix(f'.{suffix}')
            )
            out.write_bytes(data)
            out.parent.mkdir(exist_ok=True, parents=True)
            paths.append(outfl)
            return tuple(paths)

        # If there are multiple pages, write them into the parent folder of out
        for fl_name, data in pages:
            # multi_file will add a prefix name to the file
            pfx, *fl_name = fl_name.split('_', maxsplit=1)
            fl_name = fl_name[0] if fl_name else pfx
            pfx = prefix or ''
            fl_name = f'{pfx}{fl_name}'
            outfl = out / fl_name
            duplicates = 0
            while outfl.exists():
                outfl = outfl.with_stem(f'{outfl.stem}({duplicates + 1})')
                duplicates += 1
            outfl.parent.mkdir(exist_ok=True, parents=True)
            outfl.write_bytes(data)
            paths.append(outfl)

        return tuple(paths)

    def reload(self) -> None:
        """Refresh/Reload the mapseries (refresh is shadowed by Element cache control)"""
        self.elem.refresh()

    def __iter__(self) -> Iterator[MapSeries]:
        pages = range(1, self.elem.pageCount + 1)
        current = self.elem.currentPageNumber
        try:
            for page in pages:
                self.elem.currentPageNumber = page
                yield self
        finally:
            self.elem.currentPageNumber = current

    def __len__(self) -> int:
        return self.page_count

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}('
                f'current_page="{self.name or 'None'}", '
                f'page_count={self.page_count}, '
                f'parent={self.parent}'
            ')'
        )


class Bookmark(Element[mpt.Bookmark, cim.CIMBookmark, Map]):

    @property
    def description(self) -> str:
        """The description of the Bookmark."""
        return self.elem.description

    @description.setter
    def description(self, description: str) -> None:
        self.elem.description = description

    @property
    def map(self) -> Map:
        """The Map the Bookmark is associated with. (If not built by `arcpie`, Map parent will be `None`)"""
        if not self.has_map:
            raise AttributeError(f'{self} has no associated Map')
        if self.parent:
            return self.parent
        else:
            assert self.elem.map
            return Map(self.elem.map, None)

    @property
    def has_map(self) -> bool:
        """Determine if the Bookmark has an associated Map."""
        return (self.parent or self.elem.map) is not None

    @property
    def has_thumbnail(self) -> bool:
        """Determine if the Bookmark has a thubmnail."""
        return self.elem.hasThumbnail

    def update_thumbnail(self) -> None:
        """Update the thumbnail for the Bookmark."""
        self.elem.updateThumbnail()


class BookmarkMapSeries(Element[mpt.BookmarkMapSeries, cim.CIMBookmarkMapSeries, Layout]):

    @property
    def page_count(self) -> int:
        """Total number of pages in the BookmarkMapSeries."""
        return self.elem.pageCount

    def __len__(self) -> int:
        return self.page_count

    @property
    def map_frame(self) -> MapFrame:
        """The Associated MapFrame for the BookmarkMapSeries. (raises: `AttributeError`)"""
        if not self.has_map_frame:
            raise AttributeError(f'{self} has no associated MapFrame')
        assert self.elem.mapFrame
        return MapFrame(self.elem.mapFrame, self.parent)

    @property
    def has_map_frame(self) -> bool:
        """Determine if the BookmarkMapSeries has an associated MapFrame."""
        return self.elem.mapFrame is not None

    @property
    def current_bookmark(self) -> Bookmark:
        """Get the current Bookmark."""
        if not self.has_current_bookmark:
            raise AttributeError(f'{self} has no current Bookmark')
        assert self.elem.currentBookmark
        return Bookmark(self.elem.currentBookmark, self.map_frame.map)

    @property
    def has_current_bookmark(self) -> bool:
        """Determine if there is a current Bookmark set."""
        return self.elem.currentBookmark is not None

    @property
    def bookmarks(self) -> ElementList[Bookmark]:
        """ElementList of all associated Bookmarks."""
        return self._cached('bookmarks',
            lambda: ElementList(
                Bookmark(b, self.map_frame.map)
                for b in self.elem.bookmarks
            )
        )

    @property
    def current_page_name(self) -> str:
        """Get the current Bookmark name."""
        try:
            return self.elem.currentPageName
        except Exception:
            return self.current_bookmark.name

    @current_page_name.setter
    def current_page_name(self, name: str) -> None:
        self.current_page_number = self.elem.getPageNumberFromName(name)

    @property
    def current_page_number(self) -> int:
        """Get the current page number."""
        return self.elem.currentPageNumber

    @current_page_number.setter
    def current_page_number(self, number: int) -> None:
        self.elem.currentPageNumber = number

    def reload(self) -> None:
        """Reload the BookmarkMapseries. Alias of `refresh` since `refresh` is used for cache."""
        self.elem.refresh()


# Reports export only allows a subset of PDF exporting
type _RPTFormat = mpt.PDFFormat
type _RPTFormatAlt = fmts.PDF
_RPTFormatName = Literal['PDF']


class Report(Element[mpt.Report, cim.CIMReport, Project]):

    @property
    def sections(self) -> ElementList[ReportElement[Any, Any]]:
        """ElementList of all ReportSecions *and* ReportLayoutSections."""
        return self._cached('sections',
            lambda: ElementList(
                ReportElement(e, self)
                for e in self.elem.listSections()
            )
        )

    @property
    def report_sections(self) -> ElementList[ReportSection]:
        """ElementList of all ReportSecions."""
        return ElementList(
            ReportSection(sec.elem, self)
            for sec in self.sections
            if sec.mp_type_name == 'ReportSection'
        )

    @property
    def report_layout_sections(self) -> ElementList[ReportLayoutSection]:
        """ElementList of all ReportLayoutSections"""
        return ElementList(
            ReportLayoutSection(sec.elem, self)
            for sec in self.sections
            if sec.mp_type_name == 'ReportLayoutSection'
        )

    @property
    def definition_query(self) -> str:
        """SQL where clause that the report is filtered with."""
        return self.elem.definitionQuery

    @definition_query.setter
    def definition_query(self, query: str | None) -> None:
        self.elem.definitionQuery = query or ''

    @contextmanager
    def query_as(self, query: str | None):
        """Context manager for changing the query temporarily.

        Args:
            query: The SQL where clause to use as the query.

        Example:
        ```python
        >>> for q_name, sql in report_queries.items():
        ...     with report.query_as(sql):
        ...         report.export(..., out=q_name)
        ```
        """
        cur = self.definition_query
        try:
            self.definition_query = query
            yield self
        finally:
            self.definition_query = cur

    @overload
    def export(self,  # type: ignore (No Overlap?)
               format: _RPTFormat | _RPTFormatAlt | _RPTFormatName,
               *,
               out: None = None,
               antialiasing: mpt.Antialiasing | None = ...,
               report_options: mpt.ReportExportOptions | None = ...,
               custom_pages: str | None = ...,
               page_offset: int = ...,
               page_override: int = ...,
               export_pages: mpt.ReportExportPages = ...,
        ) -> tuple[bytes, ...]: ...
    @overload
    def export(self,
               format: _RPTFormat | _RPTFormatAlt | _RPTFormatName,
               *,
               out: Path | str = ...,
               antialiasing: mpt.Antialiasing | None = ...,
               report_options: mpt.ReportExportOptions | None = ...,
               custom_pages: str | None = ...,
               page_offset: int = ...,
               page_override: int = ...,
               export_pages: mpt.ReportExportPages = ...,
        ) -> tuple[Path, ...]: ...
    def export(self,
               format: _RPTFormat | _RPTFormatAlt | _RPTFormatName,
               *,
               out: Path | str | None = None,
               antialiasing: mpt.Antialiasing | None = None,
               report_options: mpt.ReportExportOptions | None = None,
               custom_pages: str | None = None,
               page_offset: int | None = None,
               page_override: int | None = None,
               export_pages: mpt.ReportExportPages = 'ALL',
        ) -> tuple[Path, ...] | tuple[bytes, ...]:
        """Export the MapSeries.

        Args:
            format: A string name for the output format or a Formatter object.
            out: An optional output file for the export.
            antialiasing: Antialiazing level for the export.
            report_options: ReportExportOptions object (additional kwargs will set this too.)
            custom_pages: An optional custom page range string for the export.
            page_offset: An optional offset on page numbers for the report.
            page_override: Override the total number of pages.
            export_pages: The pages to export (Even, Odd, All, etc.)
        """
        rpt_opts = report_options or mp.CreateExportOptions('REPORT')
        rpt_opts = cast(mpt.ReportExportOptions, rpt_opts)
        export_pages = cast(mpt.ReportExportPages, export_pages.upper())

        if export_pages == 'CUSTOM' or custom_pages:
            if not custom_pages:
                raise ValueError("`export_pages` is set to 'custom', but no `custom_pages` argument given")
            rpt_opts.customPages = custom_pages
            rpt_opts.setExportPages('CUSTOM')
        else:
            rpt_opts.setExportPages(export_pages)
        if page_offset is not None:
            rpt_opts.startingPageNumberLabelOffset = page_offset
        if page_override is not None:
            rpt_opts.totalPageNumberOverride = page_override

        display = None
        out = Path(out) if out else None

        if antialiasing:
            display = cast(mpt.DisplayOptions, mp.CreateExportOptions('DISPLAY'))
            display.setAntialiasing(antialiasing)

        if isinstance(format, fmts.Format):
            format = format.fmt

        if type(format).__name__.endswith('Format'):
            suffix = type(format).__name__[:3].lower()
            format = cast(_RPTFormat, format)

        elif isinstance(format, str):
            if format != 'PDF':
                raise ValueError(f'Report can only export to PDF, not {format}')
            # Some type forcing required here for the subset
            suffix = format[:3].lower()
            format = cast(_RPTFormat, mp.CreateExportFormat(format))

        else:
            raise ValueError(f'Unknown format {type(format)} : {format}')

        out_name = f'{self.name}.{suffix}'

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            outfl = tmp / out_name
            format.filePath = str(outfl)
            self.elem.export(
                format,
                report_export_options=rpt_opts,
                display_options=display,
            )
            pages = tuple((fl.name, fl.read_bytes()) for fl in tmp.glob('*.pdf'))

        # Return the raw page data if an outfile is not specified
        if not out:
            return tuple(page[1] for page in pages)

        paths = list[Path]()
        fl_name, data = pages[0]
        outfl = (
            out
            if out.suffix == '.pdf'
            else out.parent / fl_name
        )
        out.write_bytes(data)
        out.parent.mkdir(exist_ok=True, parents=True)
        paths.append(outfl)
        return tuple(paths)

    def open_view(self) -> None:
        """Open the Report in the active Project."""
        self.elem.openView()


class ElevationSource(Element[mpt.ElevationSource, None, Project]):

    @property
    def source(self) -> str:
        """The string representation of the Elevation data source."""
        return self.elem.dataSource

    @property
    def visible(self) -> bool:
        """Visibility state of the ElevationSource."""
        return self.elem.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self.elem.visible = visible


class Style(Element[None, None, Project]):

    def __init__(self, name: str, parent: Project):
        if name.endswith('.stylx'):
            self.fullname = name
            self._name = name.rsplit('\\', maxsplit=1)[-1].removesuffix('.stylx')
        else:
            self.fullname = self._name = name
        super().__init__(None, parent)
        self.parent = parent
        self.elem = None
        self.cache = dict[str, list[StyleItem]]()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: Never) -> Never: ...  # type: ignore (Cannot set Style name)

    def __repr__(self):
        return f'{type(self).__name__}({self.fullname})'

    def __str__(self) -> str:
        return self.fullname

    @property
    def items(self) -> ElementList[StyleItem]:
        """ElementList of StyleItems associated with the Style/stylx."""
        return self._cached('items',
            lambda: ElementList(
                StyleItem(cast(mpt.StyleItem, item), self)
                for item in self.parent.elem.listStyleItems(self.fullname)
            )
        )

    @property
    def by_name(self) -> dict[str, ElementList[StyleItem]]:
        """Group the StyleItems by name."""
        items = dict[str, ElementList[StyleItem]]()
        for item in self.items:
            name = item.name
            items.setdefault(name, ElementList())
            items[name].append(item)
        return items

    @property
    def by_tags(self) -> dict[str, ElementList[StyleItem]]:
        """Group the StyleItems by tag. (will create duplicate item refrences)"""
        items = dict[str, ElementList[StyleItem]]()
        for item in self.items:
            for tag in item.tags:
                items.setdefault(tag, ElementList())
                items[tag].append(item)
        return items

    @property
    def by_class(self) -> dict[mpt.StyleClass, ElementList[StyleItem]]:
        """Group the StyleItems by styleClass."""
        items = dict[mpt.StyleClass, ElementList[StyleItem]]()
        for item in self.items:
            cls = item.elem.styleClass
            items.setdefault(cls, ElementList())
            items[cls].append(item)
        return items

    @property
    def by_key(self) -> dict[str, StyleItem]:
        """Create a mapping of each item to its unique key."""
        return {st.key: st for st in self.items}

    def filter_by(
        self,
        key: str | None = None,
        name: str | None = None,
        style_class: mpt.StyleClass | None = None,
        tags: Iterable[str] | str | None = None,
    ) -> ElementList[StyleItem]:
        """Apply a filter to the StyleItems in the Sytle group.

        Args:
            key: A unique key to filter by.
            name: A name expression to filter by.
            style_class: A style class to filter by.
            tags: Tags to filter by.
        """
        if isinstance(tags, str):
            tags = tags.split(';')
        tags = set(tags) if tags else set()
        items = ElementList[StyleItem]()
        for item in self.items:
            if key and item.elem.key == key:
                items.append(item)
            elif name and item.elem.name == name:
                items.append(item)
            elif tags and tags.issubset(item.tags):
                items.append(item)
            elif style_class and item.elem.styleClass == style_class:
                items.append(item)
        return items


class StyleItem(Element[mpt.StyleItem, None, Style]):

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}('
                f'name={self.elem.name}, '
                f'key={self.elem.key}, '
                f'styleClass={self.elem.styleClass}, '
                f'tags={self.tags}'
            ')'
        )

    @property
    def tags(self) -> set[str]:
        """A set of tags applied to the StyleItem."""
        return set(str(self.elem.tags).split(';'))

    @property
    def key(self) -> str:
        """The uniqiue key for the StyleItem."""
        return self.elem.key

    @property
    def style(self):
        """The string representation of the parent Style. (use `parent` for `Style` object)"""
        return self.elem.style

    @property
    def style_class(self) -> mpt.StyleClass:
        """The style class of the StyleItem."""
        return self.elem.styleClass

    @property
    def category(self) -> str:
        """The category of the StyleItem."""
        return self.elem.category


class ReportElement[MPElem: mpt.ReportSection | mpt.ReportLayoutSection, CIM](Element[MPElem, CIM, Report]):

    @property
    def visible(self) -> bool:
        """Visibility state of the ReportElement."""
        return self.elem.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self.elem.visible = visible


class ReportSectionField(TypedDict):
    fieldName: str
    sortOrder: Literal['ASC', 'DESC', 'NONE']
    groupField: bool


StatisticType = Literal['COUNT', 'MEAN', 'MEDIAN', 'SUM', 'STD_DEV', 'MAX', 'MIN']


class ReportStatistic(TypedDict):
    fieldName: str
    statistic: StatisticType


class ReferenceDataSource(TypedDict):
    dataset: str
    workspace_factory: str
    connection_info: str


class ReportSection(ReportElement[mpt.ReportSection, cim.CIMReportSection]):

    @property
    def statistics(self) -> list[ReportStatistic]:
        """A list of ReportStatistic dictionaries for the ReportSection."""
        return cast(list[ReportStatistic], self.elem.statistics)

    @property
    def definition_query(self) -> str:
        """SQL where clause that filters the ReportSection."""
        return self.elem.definitionQuery

    @definition_query.setter
    def definition_query(self, query: str | None) -> None:
        self.elem.definitionQuery = query or ''

    @property
    def fields(self) -> list[ReportSectionField]:
        """A list of ReportSectionFields dictionaries for the ReportSection."""
        return cast(list[ReportSectionField], self.elem.fields)

    @fields.setter
    def fields(self, fields: Iterable[ReportSectionField]) -> None:
        self.elem.fields = list(cast(list[dict[str, str | bool]], fields))

    @property
    def source(self) -> ReferenceDataSource:
        """A ReferenceDataSource dictionary that defines the data connection."""
        return cast(ReferenceDataSource, self.elem.referenceDataSource)

    def set_source(self, source: LayerLike | TableLike | fc.FeatureClass | fc.Table | Path | str):
        """Set the ReportSection data source to something that has data.

        Args:
            source: Any valid source object (Layer, Table, FeatureClass, URL, Path, etc.)
        """
        if isinstance(source, Element):
            source = source.elem
        else:
            source = str(source)
        self.elem.setReferenceDataSource(source)

    def set_report_source(self, source: LayerLike | TableLike | fc.FeatureClass | fc.Table | Path | str, name: str | None = None, rel_class: str | None = None):
        """Set the related Report data source.

        Args:
            source: Any valid data source for a Report.
            name: The name of the datasource.
            rel_class: The name of the new relationship class.
        """
        if isinstance(source, Element):
            source = source.elem
        else:
            source = str(source)
        self.elem.setRelatedReportSource(source, name, rel_class)

    @contextmanager
    def query_as(self, query: str | None):
        """Create a query context for the ReportSection.

        Args:
            query: SQL where clause to set for the context block.

        Example:
        ```python
        >>> for q_name, sql in section_queries.items():
        ...     with report_section.query_as(sql):
        ...         report.export(..., name=q_name)
        ```
        """
        cur = self.definition_query
        try:
            self.definition_query = query
            yield self
        finally:
            self.definition_query = cur


class ReportLayoutSection(ReportElement[mpt.ReportLayoutSection, cim.CIMReportLayoutPageSection]): ...


class LayoutElement[MPElem: mpt.LayoutElement, CIM](Element[MPElem, CIM, Layout]):

    @property
    def group(self) -> GroupElement | None:
        """GroupElement the LayoutElement belongs to if it exists."""
        if self.elem.parentGroupElement:
            return GroupElement(self.elem.parentGroupElement, self.parent)

    @property
    def type(self) -> mpt.ElementType:
        """The string name for the LayoutElement type."""
        return cast(mpt.ElementType, self.elem.type)

    @property
    def visible(self) -> bool:
        """Visibility state for the LayoutElement."""
        return self.elem.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self.elem.visible = visible

    @property
    def height(self) -> float:
        """LayoutElement height in page units."""
        return self.elem.elementHeight

    @height.setter
    def height(self, height: float) -> None:
        self.elem.elementHeight = height

    @property
    def width(self) -> float:
        """LayoutElement width in page units."""
        return self.elem.elementWidth

    @width.setter
    def width(self, width: float) -> None:
        self.elem.elementWidth = width

    @property
    def anchor(self) -> mpt.Anchor:
        """LayoutElement anchor point."""
        return self.elem.anchor

    @anchor.setter
    def anchor(self, anchor: mpt.Anchor) -> None:
        self.elem.setAnchor(anchor)

    @property
    def rotation(self) -> float:
        """LayoutElement rotation in degrees. (`TableFrameElements` cannot be rotated)"""
        if not isinstance(self, TableFrameElement):
            return self.elem.elementRotation  # type: ignore
        return 0.0

    @rotation.setter
    def rotation(self, rotation: float) -> None:
        if not isinstance(self, TableFrameElement):
            self.elem.elementRotation = rotation  # type: ignore

    @property
    def x(self) -> float:
        """The X position of the LayoutElement anchor point in page units."""
        return self.elem.elementPositionX

    @x.setter
    def x(self, x: float) -> None:
        self.elem.elementPositionX = x

    @property
    def y(self) -> float:
        """The Y position of the LayoutElement anchor point in page units."""
        return self.elem.elementPositionY

    @y.setter
    def y(self, y: float) -> None:
        self.elem.elementPositionY = y

    @property
    def locked(self) -> bool:
        """The editability state of the LayoutElement."""
        return self.elem.locked

    @locked.setter
    def locked(self, locked: bool) -> None:
        self.elem.locked = locked

    def delete(self) -> None:
        """Delete the LayuoutElement from its parent Layout (if it has one)."""
        if self.parent is not None:
            self.parent.delete_element(self)

    def move(self, x: float = 0.0, y: float = 0.0) -> None:
        """Shift the element by the provided x/y deltas."""
        self.x += x
        self.y += y


class MapSurroundElement(LayoutElement[mpt.MapSurroundElement, cim.CIMMapSurround]):

    def apply_style(self, style_item: StyleItemLike) -> None:
        """Apply a `MAP_SURROUND` StyleItem to the MapSurroundElement."""
        style_item = style_item.elem if isinstance(style_item, Element) else style_item
        if style_item.styleClass != 'MAP_SURROUND':
            raise ValueError(f'{self} can only use `MAP_SURROUND` style classes, not {style_item.styleClass}')
        self.elem.applyStyleItem(style_item)


class TableFrameElement(LayoutElement[mpt.TableFrameElement, cim.CIMTableFrame]):

    def apply_style(self, style_item: StyleItemLike) -> None:
        """Apply a `TABLE_FRAME` StyleItem to the TableFrameElement."""
        style_item = style_item.elem if isinstance(style_item, Element) else style_item
        if style_item.styleClass != 'TABLE_FRAME':
            raise ValueError(f'{self} can only use `TABLE_FRAME` style classes, not {style_item.styleClass}')
        self.elem.applyStyleItem(style_item)


class GraphicElement(LayoutElement[mpt.GraphicElement, cim.CIMGraphicElement]):

    def apply_style(self, style_item: StyleItemLike) -> None:
        """Apply a `` StyleItem to the MapSurroundElement."""
        style_item = style_item.elem if isinstance(style_item, Element) else style_item
        if style_item.styleClass not in ('POINT', 'LINE', 'POLYGON'):
            raise ValueError(f'{self} can only use `POINT`, `LINE`, or `POLYGON` style classes, not {style_item.styleClass}')
        self.elem.applyStyleItem(style_item)

    def clone(self, name: str | None = None) -> Self:
        """Clone the GraphicElement with an optional new name. (default: `{name}{n+1}`)"""
        new = type(self)(self.elem.clone(), self.parent)
        new.name = name or new.name
        return new


class GroupElement(LayoutElement[mpt.GroupElement, cim.CIMGroupElement]):

    @property
    def elements(self) -> ElementList[LayoutElement[Any, Any]]:
        """ElementList of all LayoutElements within the GroupElement. (use other properties for typed/filtered LayoutElements)"""
        return self._cached('elements',
            lambda: ElementList(
                LayoutElement[Any, Any](elem, self.parent)
                for elem in self.elem.elements
        ))

    @property
    def graphic_elements(self) -> ElementList[GraphicElement]:
        """ElementList of all GraphicElements within the GroupElement."""
        return ElementList(
            GraphicElement(cast(mpt.GraphicElement, elem), self.parent)
            for elem in self.elements
            if elem.type == 'GRAPHIC_ELEMENT'
        )

    @property
    def group_elements(self) -> ElementList[GroupElement]:
        """ElementList of all GroupElements within the GroupElement."""
        return ElementList(
            GroupElement(cast(mpt.GroupElement, elem), self.parent)
            for elem in self.elements
            if elem.type == 'GROUP_ELEMENT'
        )

    @property
    def legend_elements(self) -> ElementList[LegendElement]:
        """ElementList of all LegendElements within the GroupElement."""
        return ElementList(
            LegendElement(cast(mpt.LegendElement, elem), self.parent)
            for elem in self.elements
            if elem.type == 'LEGEND_ELEMENT'
        )

    @property
    def map_frames(self) -> ElementList[MapFrame]:
        """ElementList of all MapFrames within the GroupElement."""
        return ElementList(
            MapFrame(cast(mpt.MapFrame, elem), self.parent)
            for elem in self.elements
            if elem.type == 'MAPFRAME_ELEMENT'
        )

    @property
    def map_surround_elements(self) -> ElementList[MapSurroundElement]:
        """ElementList of all MapSurroundElements within the GroupElement."""
        return ElementList(
            MapSurroundElement(cast(mpt.MapSurroundElement, elem), self.parent)
            for elem in self.elements
            if elem.type == 'MAPSURROUND_ELEMENT'
        )

    @property
    def picture_elements(self) -> ElementList[PictureElement]:
        """ElementList of all PictureElements within the GroupElement."""
        return ElementList(
            PictureElement(cast(mpt.PictureElement, elem), self.parent)
            for elem in self.elements
            if elem.type == 'PICTURE_ELEMENT'
        )

    @property
    def table_frame_elements(self) -> ElementList[TableFrameElement]:
        """ElementList of all TableFrameElements within the GroupElement."""
        return ElementList(
            TableFrameElement(cast(mpt.TableFrameElement, elem), self.parent)
            for elem in self.elements
            if elem.type == 'TABLEFRAME_ELEMENT'
        )

    @property
    def text_elements(self) -> ElementList[TextElement]:
        """ElementList of all TextElements within the GroupElement."""
        return ElementList(
            TextElement(cast(mpt.TextElement, elem), self.parent)
            for elem in self.elements
            if elem.type == 'TEXT_ELEMENT'
        )


LegendFittingStrategy = Literal[
    'AdjustFontSize',
    'AdjustColumns',
    'AdjustColumnsAndFont',
    'AdjustFrame',
    'ManualColumns',
]


class LegendElement(LayoutElement[mpt.LegendElement, cim.CIMLegend]):

    @property
    def map(self) -> Map:
        if self.has_map_frame and self.map_frame.has_map:
            return self.map_frame.map
        raise AttributeError(f'{self} has no associated Map or MapFrame')

    @property
    def map_frame(self) -> MapFrame:
        """Get the MapFrame associated with the LegendElement. (raises: `AttributeError`)"""
        if self.has_map_frame:
            assert self.elem.mapFrame
            return MapFrame(self.elem.mapFrame, self.parent)
        raise AttributeError(f'{self} has no associated MapFrame')

    @map_frame.setter
    def map_frame(self, frame: MapFrameLike) -> None:
        frame = frame.elem if isinstance(frame, Element) else frame
        self.elem.mapFrame = frame

    @property
    def has_map_frame(self) -> bool:
        """Determine if the LegendElement has an associated MapFrame."""
        return self.elem.mapFrame is not None

    @property
    def columns(self) -> int:
        """The number of columns in the Legend."""
        return self.elem.columnCount

    @columns.setter
    def columns(self, cols: int) -> None:
        self.elem.columnCount = cols

    @property
    def title(self) -> str:
        """The title of the LegendElement."""
        return self.elem.title

    @title.setter
    def title(self, title: str) -> None:
        self.elem.title = title

    @property
    def fitting_strategy(self) -> LegendFittingStrategy:
        """The fitting strategy of the LegendElement."""
        return cast(LegendFittingStrategy, self.elem.fittingStrategy)

    @fitting_strategy.setter
    def fitting_strategy(self, strategy: LegendFittingStrategy) -> None:
        self.elem.fittingStrategy = strategy

    @property
    def overflowing(self) -> bool:
        """Determine if the LegendElement is overflowing its bounds."""
        return self.elem.isOverflowing

    @property
    def show_title(self) -> bool:
        """Title visibility state of the LegendElement."""
        return self.elem.showTitle

    @show_title.setter
    def show_title(self, show: bool) -> None:
        self.elem.showTitle = show

    @property
    def sync_order(self) -> bool:
        return self.elem.syncLayerOrder

    @sync_order.setter
    def sync_order(self, sync: bool) -> None:
        """Sync legend order with Map Layer order."""
        self.elem.syncLayerOrder = sync

    @property
    def sync_visibility(self) -> bool:
        """Sync legend visibility with Map Layer visibility."""
        return self.elem.syncLayerVisibility

    @sync_visibility.setter
    def sync_visibility(self, sync: bool) -> None:
        self.elem.syncLayerVisibility = sync

    @property
    def sync_new(self) -> bool:
        """Add new Layers to the LegendElement when they are added to the Map."""
        return self.elem.syncNewLayer

    @sync_new.setter
    def sync_new(self, sync: bool) -> None:
        self.elem.syncNewLayer = sync

    @property
    def sync_scale(self) -> bool:
        """Sync the reference scale of the LegendElement with the Map reference scale."""
        return self.elem.syncReferenceScale

    @sync_scale.setter
    def sync_scale(self, sync: bool) -> None:
        self.elem.syncReferenceScale = sync

    def sync(self, *on: Literal['visibility', 'new', 'scale', 'order']) -> None:
        """Set sync properties for the LegendElement.

        Note:
            Any properties not explicitly passed to this method will be disabled.
        """
        self.sync_visibility = 'visibility' in on
        self.sync_new = 'new' in on
        self.sync_scale = 'scale' in on
        self.sync_order = 'order' in on

    @property
    def items(self) -> ElementList[LegendItem]:
        """ElementList of LegendItems in the LegendElement.

        Items can be mutated directly to update them. No need or ability to set this.
        """
        return self._cached('items',
            lambda: ElementList(
                LegendItem(itm, self)
                for itm in self.elem.items or []
            )
        )

    def add_item(self, layer: LayerLike, position: mpt.AddPosition = 'TOP') -> LegendItem:
        """Add a Layer to the Legend.

        Args:
            layer: A Layer to add to the Legend.
            position: The position to add the layer in. (default: `TOP`)
        """
        layer = layer.elem if isinstance(layer, Element) else layer
        item = LegendItem(self.elem.addItem(layer, position), self)
        self.refresh('items')
        return item

    def remove_item(self, item: LegendItemLike) -> None:
        """Remove a LegedItem from the LegendElement."""
        item = item.elem if isinstance(item, Element) else item
        self.elem.removeItem(item)
        self.refresh('items')

    def move_item(self, reference: LegendItemLike, item: LegendItemLike, position: mpt.MovePosition = 'BEFORE'):
        """Move a LegendItem to a different location in the LegendElement.

        Args:
            reference: The reference LegendItem to move the item relative to.
            item: The LegendItem to move.
            position: Where to move the item relative to the reference. (default: `BEFORE`)
        """
        reference = reference.elem if isinstance(reference, Element) else reference
        item = item.elem if isinstance(item, Element) else item
        self.elem.moveItem(reference, item, position)

    def apply_style(self, style_item: StyleItemLike):
        """Apply a `LEGEND` style item to the LegendElement."""
        style_item = style_item.elem if isinstance(style_item, Element) else style_item
        if style_item.styleClass != 'LEGEND':
            raise ValueError(f'{self} can only use `LEGEND` style classes, not {style_item.styleClass}')
        self.elem.applyStyleItem(style_item)


LegendItemArrangement = Literal[
    'PatchLabelDescription',
    'PatchDescriptionLabel',
    'LabelPatchDescription',
    'LabelDescriptionPatch',
    'DescriptionPatchLabel',
    'DescriptionLabelPatch',
]


class LegendItem(Element[mpt.LegendItem, cim.CIMLegendItem, LegendElement]):

    @property
    def arrangement(self) -> LegendItemArrangement:
        """The patch/description/label arrangement order for the LegendItem."""
        return cast(LegendItemArrangement, self.elem.arrangement)

    @arrangement.setter
    def arrangement(self, arrangement: LegendItemArrangement) -> None:
        self.elem.arrangement = arrangement

    @property
    def column(self) -> int:
        return self.elem.column

    @column.setter
    def column(self, col: int) -> None:
        self.elem.column = col

    @property
    def patch_height(self) -> float:
        """The patch height of the LayoutItem in points."""
        return self.elem.patchHeight

    @patch_height.setter
    def patch_height(self, height: float) -> None:
        self.elem.patchHeight = height

    @property
    def patch_width(self) -> float:
        """The patch width of the LayoutItem in points."""
        return self.elem.patchWidth

    @patch_width.setter
    def patch_width(self, width: float) -> None:
        self.elem.patchHeight = width

    @property
    def show_feature_count(self) -> bool:
        return self.elem.showFeatureCount

    @show_feature_count.setter
    def show_feature_count(self, show: bool) -> None:
        self.elem.showFeatureCount = show

    @property
    def show_visible(self) -> bool:
        return self.elem.showVisibleFeatures

    @show_visible.setter
    def show_visibleure_count(self, show: bool) -> None:
        self.elem.showVisibleFeatures = show

    def apply_style(self, style_item: StyleItemLike) -> None:
        """Apply a `LEGEND_ITEM` style to the LegendItem."""
        style_item = style_item.elem if isinstance(style_item, Element) else style_item
        if style_item.styleClass != 'LEGEND_ITEM':
            raise ValueError(f'{self} can only use `LEGEND_ITEM` style classes, not {style_item.styleClass}')
        self.elem.applyStyleItem(style_item)


class PictureElement(LayoutElement[mpt.PictureElement, cim.CIMPictureGraphic]):

    @property
    def alt_text(self) -> str:
        """The hover text of the PictureElement."""
        return self.elem.altText

    @alt_text.setter
    def alt_text(self, text: str) -> None:
        self.elem.altText = text

    @property
    def image_data(self) -> bytes:
        """Get the raw bytes for the source image. Will work with URLs and local filepaths."""
        try:
            if self.source.startswith('http'):
                return httpx.get(self.source).read()
            elif Path(self.source).exists():
                return Path(self.source).read_bytes()
            else:
                raise FileNotFoundError('Cannot Resolve file with URL or filepath, Trying to read form aprx archive...')
        except Exception:
            try:
                cim_path: str = self.cim_dict['graphic']['referenceURI'][8:]
                cim_path = cim_path.replace('\\', '/')
                assert self.parent
                assert self.parent.parent
                return self.parent.parent.read_cim_file(cim_path)
            except Exception as e2:
                raise AttributeError(f'{self.source} cannot be resolved.') from e2

    @property
    def source(self) -> str:
        """A string representation of the source image path. (filepath or URL)"""
        return self.elem.sourceImage

    @source.setter
    def source(self, image: Path | str) -> None:
        self.elem.sourceImage = str(image)


class TextElement(LayoutElement[mpt.TextElement, cim.CIMGraphicElement]):

    @property
    def font_family(self) -> str:
        """The Font Family of the Text Element. (must be proper name of an installed font)"""
        return self.elem.fontFamilyName

    @font_family.setter
    def font_family(self, family: str) -> None:
        self.elem.fontFamilyName = family

    @property
    def font_style(self) -> str:
        """The style name of the font (bold, italic, regular, etc.). Must be supported by font family."""
        return self.elem.fontStyleName

    @font_style.setter
    def font_style(self, style: str) -> None:
        self.elem.fontStyleName = style

    @property
    def overflowing(self) -> bool:
        """Determine if the TextElement text is overflowing the TextElement boundary."""
        return self.elem.isOverflowing

    @property
    def text(self) -> str:
        """The text string of the TextElement."""
        return self.elem.text

    @text.setter
    def text(self, text: str) -> None:
        self.elem.text = text

    @property
    def text_angle(self) -> float:
        """The rotation angle of the text in degrees."""
        return self.elem.textAngle

    @text_angle.setter
    def text_angle(self, angle: float) -> None:
        self.elem.textAngle = angle

    @property
    def text_size(self) -> float:
        """The text size in points."""
        return self.elem.textSize

    @text_size.setter
    def text_size(self, size: float) -> None:
        """The text size in points."""
        self.elem.textSize = size

    def set_text(self,
                 *,
                 text: str | None = None,
                 family: str | None = None,
                 style: str | None = None,
                 size: float | None = None,
                 angle: float | None = None,
    ) -> None:
        """Set properties of the TextElement text.

        Args:
            text: Set the text.
            family: Set the font family.
            style: Set the font style.
            size: Set the font size.
            angle: Set the font angle.
        """
        if text is not None:
            self.text = text
        if family is not None:
            self.font_family = family
        if style is not None:
            self.font_style = style
        if size is not None:
            self.size = size
        if angle is not None:
            self.angle = angle

    def apply_style(self, style_item: StyleItemLike) -> None:
        """Apply a `TEXT` StyleItem to the TextElement."""
        style_item = style_item.elem if isinstance(style_item, Element) else style_item
        if style_item.styleClass != 'TEXT':
            raise ValueError(f'{self} can only use `TEXT` style classes, not {style_item.styleClass}')
        self.elem.applyStyleItem(style_item)

    def clone(self, name: str | None = None) -> Self:
        """Make a copy of the TextElement in the parent Layout.

        Args:
            name: An optional name to give the copy. (default: `{name}{n+1}`)
        """
        new = type(self)(self.elem.clone(), self.parent)
        new.name = name or new.name
        return new
