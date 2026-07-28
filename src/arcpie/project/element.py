from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from copy import copy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    SupportsIndex,
    cast,
    overload,
)
from zipfile import ZIP_DEFLATED, ZipFile

import arcpy._mp as mpt
import arcpy.cim as cim
import arcpy.mp as mp
from arcpy import SpatialReference
from arcpy.cim.cimloader import jsontocim
from arcpy.cim.cimloader.cimtojson import CimJsonEncoder as CIMJsonEncoder

# Remove after testing
try:
    from ..database import Dataset
except ImportError:
    import sys
    root = str(Path(__file__).parent.parent.parent.resolve())
    sys.path.append(root)

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
        {'__getattr__': lambda s, n: None}
    )()


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
    | None
)


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
        or (lambda: f'ID:{hex(id(elem))}')
    )


def _get_uri(elem: MPElement) -> str:
    """Try all possible ways to get a URI from a mapping element.
     Fallback of ``NO_URI:{type(elem)}<{get_name(elem)}@{hex(id(elem))}>`` is used if uRI cannot be determined.
    """
    props = _get_props(elem, 'URI')
    fallback_uri = f'NO_URI:{type(elem)}<{_get_name(elem)}@{hex(id(elem))}>'
    if (uri := props.get('URI')):
        return uri
    if not (arc_object := getattr(elem, '_arc_object', None)):
        return fallback_uri
    if not (cim_str := getattr(arc_object, 'GetCimJSONString', None)):
        return fallback_uri
    cim: dict[str, Any] | None = json.loads(cim_str() or '{}')
    if not isinstance(cim, dict) or 'uRI' not in cim:
        return fallback_uri
    return str(cim['uRI'])


# Element takes Base arcpy.mp element, cim definition (from `getDefinition`) and parent type
class Element[MPElem: MPElement, CIMDef, Parent: Element[Any, Any] | None = None]:
    def __init__(self, elem: MPElem, parent: Parent | None = None) -> None:
        self.elem = elem
        self.__elemattrs = set(dir(elem))
        self.parent = parent
        self.type = type(elem)
        self.name = _get_name(elem)
        self.uri = _get_uri(elem)
        self.unique_name = f'{self.name}:{self.uri}'
        self.cache = dict[str, Any]()

    def _cached[T](self, attr: str, default: Callable[[], T] | None = None) -> T:
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
            return
        for prop in props:
            self.cache.pop(prop, None)

    @property
    def cim(self) -> CIMDef:
        return self.elem.getDefinition('V3')  # type: ignore

    @property
    def cim_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.cim or '{}', cls=CIMJsonEncoder))

    def __getattr__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name in self.__elemattrs:
                return getattr(self.elem, name)
            raise

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.name})'

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) or (isinstance(other, type(self)) and self.elem == other.elem)

    @classmethod
    def diff(cls, a: Self, b: Self, *, outfile: Path | str | None = None) -> str:
        """Generate a diff of two Layers using `cim_dict` and `difflib.unified_diff`"""
        diff = '\n'.join(difflib.unified_diff(
            json.dumps(a.cim_dict, indent=2).split('\n'),
            json.dumps(b.cim_dict, indent=2).split('\n'),
            fromfile=f'{a.name} (a)',
            tofile=f'{b.name} (b)',
        ))
        if outfile:
            Path(outfile).with_suffix('.diff').write_text(diff)
        return diff


class ElementList[E: Element[Any, Any, Any]](list[E]):
    """Simple list wrapper that allows accessing elements from a list by name/uRI."""

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> E: ...
    @overload
    def __getitem__(self, s: slice, /) -> list[E]: ...
    @overload
    def __getitem__(self, key: str, /) -> list[E]: ...
    @overload
    def __getitem__(self, key: re.Pattern[str], /) -> list[E]: ...
    def __getitem__(self, key: SupportsIndex | slice | str | re.Pattern[str]) -> E | list[E]:
        if isinstance(key, (str, re.Pattern)):
            if matches := [e for e in self if e.name == key or e.uri == key]:
                return matches
            if matches := [e for e in self if re.search(key, e.name)]:
                return matches
            raise IndexError(f'No elements with name {key} found')
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        try:
            self[key]
            return True
        except IndexError:
            return False

    def filter(self, cond: Callable[[E], bool]) -> list[E]:
        """Filter elements in the list using the provided function"""
        return [e for e in self if cond(e)]

    def get[D](self, key: str, default: D = None, /) -> list[E] | D:
        """Only works when indexing the list with a string name"""

        try:
            return self[key]
        except IndexError:
            return default

    def copy(self) -> ElementList[E]:
        return type(self)(super().copy())


class Project(Element[mpt.ArcGISProject, cim.CIMGISProject]):
    """Project"""

    # Need to override Element.__init__ since ArcGISProject objects are special
    def __init__(self, aprx: Path | str | None) -> None:
        super().__init__(
            mp.ArcGISProject(str(aprx) if aprx else 'CURRENT'), None
        )
        self.name = self.path.name

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.path})'

    # ArcGISProject CIM is not directly available and needs to be loaded from
    # the raw GISProject.json file in the aprx zip directory
    @property
    def cim(self) -> cim.CIMGISProject:
        with ZipFile(self.path) as zf, zf.open('GISProject.json') as cim:
            return jsontocim.GetJSONTypeOBJ(json.load(cim))  # type: ignore

    @property
    def path(self) -> Path:
        return Path(self.elem.filePath)

    @property
    def home(self) -> Path:
        return Path(self.elem.homeFolder)

    @property
    def maps(self) -> ElementList[Map]:
        return self._cached('maps',
            lambda: ElementList(
                Map(map, self)
                for map in self.elem.listMaps()
            )
        )

    @property
    def layouts(self) -> ElementList[Layout]:
        return self._cached('layouts',
            lambda: ElementList(
                Layout(layout, self)
                for layout in self.elem.listLayouts()
            )
        )

    @property
    def reports(self) -> ElementList[Report]:
        return self._cached('reports',
            lambda: ElementList(
                Report(report, self)
                for report in self.elem.listReports()
            )
        )

    @property
    def styles(self) -> ElementList[Style]:
        return self._cached('styles',
            lambda: ElementList(
                Style(st, self)
                for st in self.elem.styles
            )
        )

    @property
    def active_map(self) -> Map | None:
        active = self.elem.activeMap
        if not active:
            return None
        return Map(active, self)

    @property
    def active_view(self) -> MapView | Layout | Report | None:
        active = self.elem.activeView
        view_type = type(active).__name__
        if view_type == 'MapView':
            return MapView(cast(mpt.MapView, active), self)
        if view_type == 'Layout':
            return Layout(cast(mpt.Layout, active), self)
        if view_type == 'Report':
            return Report(cast(mpt.Report, active), self)

    @property
    def databases(self) -> list[Dataset]:
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
        return db.Dataset(self.elem.defaultGeodatabase)

    def save(self) -> None:
        """Save the project

        Raises:
            ``PermissionError`` if the file is ReadOnly
        """
        if self.elem.isReadOnly:
            raise PermissionError(f'{self.name} is read only!')
        self.elem.save()

    def save_as(self, path: Path | str) -> Project:
        """Save a copy of the project"""
        path = Path(path).with_suffix('.aprx')
        self.elem.saveACopy(str(path))
        return type(self)(path)

    def start(self) -> None:
        """Open the project using ``os.startfile``"""
        os.startfile(self.path)  # noqa: S606

    def to_directory(self, to: Path | str, *, overwrite: bool = False) -> Path:
        """Unzip the aprx file to a target directory"""
        to = Path(to).resolve()
        if to.exists() and not overwrite:
            raise FileExistsError(f'{to} exists and `overwrite` is set to `False`')
        with ZipFile(self.path) as zf:
            zf.extractall(to)
        return to

    def delete(self, recursive: bool = False) -> None:
        """Delete the project

        Args:
            recursive: If set, recursively delete the homeFolder (default: `False`)
        """
        home_folder = self.elem.homeFolder
        aprx = self.elem.filePath
        if recursive:
            shutil.rmtree(home_folder)
        else:
            Path(aprx).unlink()

    def import_document(self, doc: Path | str, *, include_layout: bool = True, reuse_existing_maps: bool = True) -> Layout | Map | Report:
        """Import a document file into this project using `ArcGISProject.importDocument`.

        Args:
            doc: The path to the document (`.pagx`, `.mapx`, `.rptx`, `.mxd`, ...[see `importDocument`])
            include_layout: Include layouts with `.mapx` files
            reuse_existing_maps: Reuse existing maps with `.pagx` files

        Returns:
            Layout | Map | Report: The imported object
        """
        doc = Path(doc)
        imported: Any = self.elem.importDocument(
            str(doc),
            include_layout=include_layout,
            reuse_existing_maps=reuse_existing_maps,
        )
        match imported:
            case mp.Map():
                self.refresh('maps')
                return Map(imported, parent=self)
            case mp.Layout():
                self.refresh('layouts')
                return Layout(imported, parent=self)
            case mp.Report():
                self.refresh('reports')
                return Report(imported, parent=self)
            case _:
                raise ValueError(f'Document type {doc.suffix} cannot be imported')

    def import_pagx(self, pagx: Path | str, *, reuse_existing_maps: bool = False) -> Layout:
        """Import a `.pagx` file. (see: `Project.import_document`)"""
        pagx = Path(pagx)
        if pagx.suffix != '.pagx':
            raise ValueError(f'Expected .pagx file, got {pagx.suffix}')
        imported = self.import_document(pagx, reuse_existing_maps=reuse_existing_maps)
        assert isinstance(imported, Layout)
        return imported

    def import_mapx(self, mapx: Path | str) -> Map:
        """Import a `.mapx` file. (see: `Project.import_document`)"""
        mapx = Path(mapx)
        if mapx.suffix != '.mapx':
            raise ValueError(f'Expected .mapx file, got {mapx.suffix}')
        imported = self.import_document(mapx)
        assert isinstance(imported, Map)
        return imported

    def import_mxd(self, mxd: Path | str, *, include_layout: bool = True) -> Map:
        """Import a `.mxd` file. (see: `Project.import_document`)"""
        mxd = Path(mxd)
        if mxd.suffix != '.mxd':
            raise ValueError(f'Expected .mxd file, got {mxd.suffix}')
        imported = self.import_document(mxd, include_layout=include_layout)
        assert isinstance(imported, Map)
        return imported

    def import_rptx(self, rptx: Path | str) -> Report:
        """Import a `.rptx` file. (see: `Project.import_document`)"""
        rptx = Path(rptx)
        if rptx.suffix != '.rptx':
            raise ValueError(f'Expected .rptx file, got {rptx.suffix}')
        imported = self.import_document(rptx)
        assert isinstance(imported, Report)
        return imported

    @classmethod
    def from_directory(cls, directory: Path | str, *, outfile: Path | str) -> Project:
        """Create an aprx file from a previously unzipped directory (see: `Project.to_directory`)"""
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
        home_folder: str | None = None,
        default_database: str | None = None,
        default_toolbox: str | None = None,
        create_parents: bool = True,
        overwrite: bool = False,
    ) -> Self:
        path = Path(path)
        path.mkdir(exist_ok=overwrite, parents=create_parents)
        aprx = mp.CreateArcGISProject(
            project_path=str(path),
            project_name=name,
            create_parent_folder=True,
            home_folder=home_folder,
            default_database=default_database,
            default_toolbox=default_toolbox,
        )
        return cls(aprx.filePath)


class Map(Element[mpt.Map, cim.CIMMap, Project]):

    @property
    def reference(self) -> SpatialReference:
        # Get custom ref or return default WGS84/4326
        return self.elem.spatialReference or SpatialReference('GCS_WGS_1984')

    @reference.setter
    def reference(self, reference: SpatialReference) -> None:
        self.elem.spatialReference = reference

    @property
    def scale(self) -> float:
        return self.elem.referenceScale

    @scale.setter
    def scale(self, scale: float) -> None:
        self.elem.referenceScale = scale

    @property
    def camera(self) -> mp.Camera:
        return self.elem.defaultCamera

    @camera.setter
    def camera(self, camera: mp.Camera) -> None:
        self.elem.defaultCamera = camera

    @property
    def units(self) -> str:
        return self.elem.mapUnits

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
            GroupLayer(lay, self)
            for lay in self.elem.listLayers()
            if lay.isGroupLayer
        )
    )

    @property
    def mapx(self) -> bytes:
        with tempfile.TemporaryDirectory(suffix=self.name) as tmp:
            mapx = Path(tmp) / f'{self.name}.mapx'
            self.elem.exportToMAPX(str(mapx))
            return mapx.read_bytes()

    @overload
    def add_layer(self, layer: Layer | mpt.Layer,
                  *,
                  before: Layer | mpt.Layer | None = ...,
                  after: Layer | mpt.Layer | None = ...,
                  position: mpt.AddPosition = ...,
                  group: GroupLayer | mpt.Layer | None = ...) -> Layer: ...
    @overload
    def add_layer(self, layer: GroupLayer,
                  *,
                  before: Layer | mpt.Layer | None = ...,
                  after: Layer | mpt.Layer | None = ...,
                  position: mpt.AddPosition = ...,
                  group: GroupLayer | mpt.Layer | None = ...) -> GroupLayer: ...
    def add_layer(self, layer: Layer | GroupLayer | mpt.Layer,
                  *,
                  before: Layer | mpt.Layer | None = None,
                  after: Layer | mpt.Layer | None = None,
                  position: mpt.AddPosition = 'AUTO_ARRANGE',
                  group: GroupLayer | mpt.Layer | None = None) -> Layer | GroupLayer:
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
        if isinstance(layer, (Layer, GroupLayer)):
            layer = layer.elem
        if isinstance(group, GroupLayer):
            group = group.elem
        if isinstance(before, Layer):
            before = before.elem
        if isinstance(after, Layer):
            after = after.elem

        if before:
            elem = self.elem.insertLayer(before, layer, 'BEFORE')
        elif after:
            elem = self.elem.insertLayer(after, layer, 'AFTER')
        elif group:
            elem = self.elem.addLayerToGroup(group, layer, position)
        else:
            elem = self.elem.addLayer(layer, position)

        if not isinstance(elem, mp.Layer):
            raise ValueError(f'Expected Layer object, got {type(elem)}. use `add_table` to add Tables.')

        if not is_group:
            self.refresh('layers')
            return Layer(elem, self)
        else:
            self.refresh('group_layers')
            return GroupLayer(elem, self)

    def add_table(self, table: Table | mpt.Table,
                  *,
                  group: GroupLayer | mpt.Layer | None = None) -> Table:
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

    def create_group(self, name: str,
                     *,
                     parent: GroupLayer | mpt.Layer | None = None) -> GroupLayer:
        """Create a new GroupLayer in the Map.

        Args:
            name: The name of the new GroupLayer
            parent: An optional parent group for the new layer
        """
        if parent is not None:
            if isinstance(parent, GroupLayer):
                parent = parent.elem
        return GroupLayer(self.elem.createGroupLayer(name, parent), self)

    def clear_selection(self) -> None:
        """Clear all selections in the Map."""
        self.elem.clearSelection()

    # def update_connection(self, ) -> None:
    #     self.elem.updateConnectionProperties(...)

    def clip_to(self, layer: Layer | mpt.Layer, selected: bool = False) -> None:
        """Clip all layers in the map to the footprint of the input layer

        Args:
            layer: The Layer to clip to
            selected: Clip only to the selected features (default: `False`)
        """
        if isinstance(layer, Layer):
            layer = layer.elem
        self.elem.clipLayers(layer, selection='SELECTED' if selected else 'ALL')

    def add_data(self, path: Path | str, service_type: mpt.WebServiceType = 'AUTOMATIC', **params: Any) -> Layer | Table:
        """Add data to the map from a path or URL.

        Args:
            path: A filepath or URL for the data element
            service_type: An optional agument for web service type (default: `AUTOMATIC`)
            **params: Additional keyword parameters passed to the webservice (optional)
        """
        elem = self.elem.addDataFromPath(str(path), web_service_type=service_type, custom_parameters=params or None)
        if isinstance(elem, mp.Layer):
            return Layer(elem, self)
        if isinstance(elem, mp.Table):
            return Table(elem, self)

        # Unreachable ? (at least not documented as reachable...)
        raise ValueError(f'Something went wrong got {type(elem)} but expected Layer or Table')

    def export_mapx(self, outfile: Path | str) -> Path:
        outfile = Path(outfile)
        with outfile.open('wb') as mapx:
            mapx.write(self.mapx)
        return outfile

    def filter(self, pred: Callable[[Layer], bool]) -> ElementList[Layer]:
        return ElementList(lay for lay in self.layers if pred(lay))


class MapView(Element[mpt.MapView, cim.CIMMapView, Project]):

    @property
    def camera(self) -> mp.Camera:
        return self.elem.camera

    @camera.setter
    def camera(self, camera: mp.Camera) -> None:
        self.elem.camera = camera

    @property
    def map(self) -> Map:
        return Map(self.elem.map, self.parent)

    @overload
    def export(self,  # type: ignore (No Overlap?)
               format: mp.Format | mpt.ExportFormat,
               *,
               outfile: None = None,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> bytes: ...
    @overload
    def export(self,
               format: mp.Format | mpt.ExportFormat,
               *,
               outfile: Path | str = ...,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> Path: ...
    def export(self,
               format: mp.Format | mpt.ExportFormat,
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
            out = tmp / f'{self.name}_export.{suffix}'
            format.filePath = str(out)
            self.elem.export(format, display_options=display)
            data = out.read_bytes()
        if outfile:
            outfile.parent.mkdir(exist_ok=True, parents=True)
            outfile.with_suffix(suffix).write_bytes(data)
            return outfile
        else:
            return data


class GroupLayer(Element[mpt.Layer, cim.CIMGroupLayer, Map]):

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

    def add_layer(self, layer: Layer | mpt.Layer, position: mpt.AddPosition = 'AUTO_ARRANGE') -> Layer:
        if not self.parent:
            raise ValueError(f'{self} is unbound and cannot add new layers')
        if isinstance(layer, Layer):
            layer = layer.elem
        return self.parent.add_layer(layer, position=position, group=self)

    def export_lyrx(self, outdir: Path | str, *, name: str | None = None, indent: int = 2) -> Path:
        outdir = Path(outdir)
        name = name or self.name
        outdir = outdir / name
        outdir = outdir.with_suffix('.lyrx')
        outdir.parent.mkdir(exist_ok=True, parents=True)
        outdir.write_text(json.dumps(self.lyrx, indent=indent))
        return outdir


class Layer(Element[mpt.Layer, cim.CIMBaseLayer, Map | GroupLayer]):

    @property
    def feature_class(self) -> fc.FeatureClass:
        """Get the associated FeatureClass object for the layer

        Raises:
            ConnectionError: If the layer has no FeatureClass (e.g. Raster/TileLayer)
        """
        # Access the fields to force an exception if the layer has no featureclass
        try:
            feature_class = fc.FeatureClass.from_layer(self.elem)
        except (RuntimeError, AttributeError) as exc:
            raise ConnectionError(f'{self.name} has no associated FeatureClass') from exc
        return feature_class

    @property
    def has_feature_class(self) -> bool:
        """Check to see if the Layer has a valid FeatureClass association"""
        try:
            _ = self.feature_class
            return True
        except ConnectionError:
            return False

    @property
    def cim_dict(self) -> dict[str, Any]:
        cim_dict = super().cim_dict
        # Convert absolute paths to relative databases into relative paths
        # This allows sharing layers bw projects with the same structure
        if (
            (ft := cim_dict.get('featureTable'))
            and (conn := ft.get('dataConnection'))
            and (ws_conn := conn.get('workspaceConnectionString'))
            and (fc := self.feature_class)
        ):
            cur_path = Path(ws_conn.replace('DATABASE=', ''))
            database = Path(fc.workspace)
            if cur_path.is_absolute() and cur_path.is_relative_to(database.parent):
                conn['workspaceConnectionString'] = f'DATABASE={cur_path.relative_to(database.parent)}'
        return cim_dict

    @property
    def lyrx(self) -> dict[str, Any]:
        lyrx = dict[str, Any]()
        lyrx['type'] = 'CIMLayerDocument'
        lyrx['layers'] = [self.uri]
        lyrx['layerDefinitions'] = [self.cim_dict]
        return lyrx

    def export_lyrx(self, outdir: Path | str, *, name: str | None = None, indent: int = 2) -> Path:
        outdir = Path(outdir)
        name = name or self.name
        outdir = outdir / name
        outdir = outdir.with_suffix('.lyrx')
        outdir.parent.mkdir(exist_ok=True, parents=True)
        outdir.write_text(json.dumps(self.lyrx, indent=indent))
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


class Table(Element[mpt.Table, cim.CIMFeatureTable, Map | GroupLayer]):

    @property
    def table(self) -> fc.Table:
        return fc.Table.from_layer(cast(mpt.Layer, self.elem))

    @property
    def cim_dict(self) -> dict[str, Any]:
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
        lyrx = dict[str, Any]()
        lyrx['type'] = 'CIMLayerDocument'
        lyrx['tables'] = [self.uri]
        lyrx['layerDefinitions'] = [self.cim_dict]
        return lyrx

    def export_lyrx(self, outdir: Path | str, *, name: str | None = None, indent: int = 2) -> Path:
        outdir = Path(outdir)
        name = name or self.name
        outdir = outdir / name
        outdir = outdir.with_suffix('.lyrx')
        outdir.parent.mkdir(exist_ok=True, parents=True)
        outdir.write_text(json.dumps(self.lyrx, indent=indent))
        return outdir

    def export_csv(self, outdir: Path | str,
                   *,
                   name: str | None = None, fields: Iterable[str] | None = None, sep: str = ',', newline: str = '\n') -> Path:
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


class ElevationSurface(Element[mpt.ElevationSurface, cim.CIMLayerElevationSurface, Map]): ...


class Layout(Element[mpt.Layout, cim.CIMLayout, Project]):

    @overload
    def export(self,  # type: ignore (No Overlap?)
               format: mp.Format | mpt.ExportFormat,
               *,
               outfile: None = None,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> bytes: ...
    @overload
    def export(self,
               format: mp.Format | mpt.ExportFormat,
               *,
               outfile: Path | str = ...,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> Path: ...
    def export(self,
               format: mp.Format | mpt.ExportFormat,
               *,
               outfile: Path | str | None = None,
               antialiasing: mpt.Antialiasing | None = None,
        ) -> Path | bytes:
        """Export the Layout. (wrapper for `MapView.export`)

        Args:
            format: A Format object or a string format (string option will use defaults)
            outfile: An optional output file location to use (will override format filePath)
            antialiasing: Antialiasing options for the output file
        """
        return MapView.export(
            cast(MapView, self),
            format=format,
            outfile=outfile,
            antialiasing=antialiasing,
        )


class MapFrame(Element[mpt.MapFrame, cim.CIMMapFrame, Layout]):

    @property
    def map(self) -> Map:
        map = self.elem.map
        layout = self.parent
        if map and layout:
            return Map(map, layout.parent)
        raise ValueError(f'MapFrame {self.name} has no associated Map')

    @property
    def has_map(self) -> bool:
        """Check if the MapFrame has an associated Map"""
        return self.elem.map is not None

    @overload
    def export(self,  # type: ignore (No Overlap?)
               format: mp.Format | mpt.ExportFormat,
               *,
               outfile: None = None,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> bytes: ...
    @overload
    def export(self,
               format: mp.Format | mpt.ExportFormat,
               *,
               outfile: Path | str = ...,
               antialiasing: mpt.Antialiasing | None = ...,
        ) -> Path: ...
    def export(self,
               format: mp.Format | mpt.ExportFormat,
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
        return MapView.export(
            cast(MapView, self),
            format=format,
            outfile=outfile,
            antialiasing=antialiasing,
        )


class MapSeries(Element[mpt.MapSeries, cim.CIMMapSeries, Layout]):

    @property
    def enabled(self) -> bool:
        return self.elem.enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self.elem.enabled = enabled

    @property
    def clip_to_features(self) -> bool:
        return self.elem.clipToIndexFeature

    @clip_to_features.setter
    def clip_to_features(self, clip_to_features: bool) -> None:
        self.elem.clipToIndexFeature = clip_to_features

    @property
    def map_frame(self) -> MapFrame:
        return MapFrame(self.elem.mapFrame, self.parent)

    @property
    def layer(self) -> Layer:
        return Layer(self.elem.indexLayer, self.map_frame.map)

    @property
    def page_name_field(self) -> str:
        return self.elem.pageNameField.name

    def reload(self) -> None:
        """Refresh/Reload the mapseries (refresh is shadowed by Element cache control)"""
        self.elem.refresh()

    @property
    def current_page_name(self) -> int | str:
        try:
            return self.elem.currentPageName
        except Exception:
            # <=3.6 compat
            return cast(int | str, self.page_row[self.page_name_field])

    @property
    def current_page_number(self) -> int | str:
        return self.elem.currentPageNumber

    @property
    def page_numbers(self) -> list[int]:
        return self.elem.selectedIndexFeatures

    @property
    def features(self) -> dict[int, dict[str, Any]]:
        return self._cached('features',
            lambda: {
                page: layout.page_row
                for page, layout in zip(self.page_numbers, self, strict=True)
        })

    @property
    def page_row(self) -> dict[str, Any]:
        """Return a dictionary representing the current map page feature"""
        pr = cast(Any, self.elem.pageRow)
        if isinstance(pr, tuple):
            return pr._asdict()  # type: ignore (NamedTuple)
        if isinstance(pr, dict):
            return cast(dict[str, Any], pr)
        else:
            return pr

    def __iter__(self) -> Iterator[MapSeries]:
        assert self.parent
        pages = self.elem.selectedIndexFeatures
        current = self.elem.currentPageNumber
        try:
            for page in pages:
                self.elem.currentPageNumber = page
                yield self
        finally:
            self.elem.currentPageNumber = current


class Bookmark(Element[mpt.Bookmark, cim.CIMBookmark, MapFrame]): ...


class BookmarkMapSeries(Element[mpt.BookmarkMapSeries, cim.CIMBookmarkMapSeries, Layout]): ...


class Report(Element[mpt.Report, cim.CIMReport, Project]): ...


class ElevationSource(Element[mpt.ElevationSource, None, Project]): ...


class Style(Element[None, None, Project]):

    def __init__(self, name: str, parent: Project):
        if name.endswith('.stylx'):
            self.fullname = name
            self.name = name.rsplit('\\', maxsplit=1)[-1].removesuffix('.stylx')
        else:
            self.fullname = self.name = name
        self.parent = parent
        self.elem = None
        self.cache = dict[str, list[StyleItem]]()

    @overload
    def __getitem__(self, key: int) -> StyleItem: ...
    @overload
    def __getitem__(self, key: str) -> list[StyleItem]: ...
    def __getitem__(self, key: int | str | Any) -> StyleItem | list[StyleItem]:
        if isinstance(key, str):
            return [s for s in self.items if s.key == key or s.name == key]
        return self.items[key]

    def __repr__(self):
        return f'{type(self).__name__}({self.fullname})'

    @property
    def items(self) -> ElementList[StyleItem]:
        return self._cached('items',
            lambda: ElementList(
                StyleItem(cast(mpt.StyleItem, item), self)
                for item in self.parent.elem.listStyleItems(self.fullname)
            )
        )

    @property
    def by_name(self) -> dict[str, ElementList[StyleItem]]:
        """Group the StyleItems by name"""
        items = dict[str, ElementList[StyleItem]]()
        for item in self.items:
            name = item.name
            items.setdefault(name, ElementList())
            items[name].append(item)
        return items

    @property
    def by_tags(self) -> dict[str, ElementList[StyleItem]]:
        """Group the StyleItems by tag (will create duplicate item refrences)"""
        items = dict[str, ElementList[StyleItem]]()
        for item in self.items:
            for tag in item.tags:
                items.setdefault(tag, ElementList())
                items[tag].append(item)
        return items

    @property
    def by_class(self) -> dict[mpt.StyleClass, ElementList[StyleItem]]:
        """Group the StyleItems by styleClass"""
        items = dict[mpt.StyleClass, ElementList[StyleItem]]()
        for item in self.items:
            cls = item.elem.styleClass
            items.setdefault(cls, ElementList())
            items[cls].append(item)
        return items

    @property
    def by_key(self) -> dict[str, StyleItem]:
        """Create a mapping of each item to its unique key"""
        return {st.key: st for st in self.items}

    def filter_by(
        self,
        key: str | None = None,
        name: str | None = None,
        style_class: mpt.StyleClass | None = None,
        tags: Iterable[str] | str | None = None,
    ) -> ElementList[StyleItem]:
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
        return set(str(self.elem.tags).split(';'))


# Remove after testing
if __name__ == '__main__':
    prj = Project(r"C:\Users\hwelch\Desktop\Louetta 8.20\Louetta 8.20.aprx")
    p1 = Project(r"C:\Users\hwelch\Desktop\Louetta 8.21\1.aprx")
    p2 = Project(r"C:\Users\hwelch\Desktop\Louetta 8.21\2.aprx")
