from __future__ import annotations

import json
import re
from collections import UserString
from collections.abc import (
    Iterable,
    Iterator,
)
from functools import cached_property
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    Unpack,
    overload,
)

# Shadow all _mp types with interface Wrappers
from arcpy._mp import (
    Bookmark as _Bookmark,  # noqa: PLC2701
    BookmarkMapSeries as _BookmarkMapSeries,  # noqa: PLC2701
    ElevationSurface as _ElevationSurface,  # noqa: PLC2701
    Layer as _Layer,  # noqa: PLC2701
    Layout as _Layout,  # noqa: PLC2701
    Map as _Map,  # noqa: PLC2701
    MapSeries as _MapSeries,  # noqa: PLC2701
    Report as _Report,  # noqa: PLC2701
    Table as _Table,  # noqa: PLC2701
)
from arcpy._symbology import Symbology
from arcpy.cim import (
    CIMBaseLayer,
    CIMBookmark,
    CIMBookmarkMapSeries,
    CIMDefinition,
    CIMElevationSurfaceLayer,
    CIMLayout,
    CIMMapDocument,
    CIMMapSeries,
    CIMReport,
    CIMStandaloneTable,
)
from arcpy.cim.cimloader.cimtojson import CimJsonEncoder
from arcpy.mp import ArcGISProject, LayerFile

from arcpie.featureclass import (
    FeatureClass,
    Table as DataTable,  # Alias Table to prevent conflict with Mapping Table
)
from arcpie.schema import SchemaLayer
from arcpie.types import (
    MapseriesPDFDefault,
    MapSeriesPDFSetting,
    PDFDefault,
    PDFSetting,
)

_MapType = TypeVar('_MapType', _Map, _Layout, _Layer, _Table, _Report, _MapSeries, _BookmarkMapSeries, _Bookmark, _ElevationSurface)
_CIMType = TypeVar('_CIMType', CIMDefinition, Any)
_Default = TypeVar('_Default')


# String Wrapper to make wildcards clear
# Since the return type of a wildcard index is different from a string index
class Wildcard(UserString):
    """Clarify that the string passed to a Manager index is a wildcard so the type checker knows you're getting a Sequence back"""


INVALID_CHARS = {'/', r'\\', ':', '*', '?', '"', '<', '>', '|'}


def safe_name(name: str) -> str:
    """Remove invalid filename characters from a string"""
    return ''.join(c for c in name if c not in INVALID_CHARS)


class MappingWrapper[MapType: (_Map, _Layout, _Layer, _Table, _Report, _MapSeries, _BookmarkMapSeries, _Bookmark, _ElevationSurface), CIMType: (CIMDefinition, Any)]:
    """Internal wrapper class for wrapping existing objects with new functionality

    Usage:
        ```python
        >>> MappingWraper[mp.<type>, cim.<type>](mp.<type>)
        ```
    """
    def __init__(self, obj: MapType, parent: _MappingObject | Project | None = None) -> None:
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
    def cim(self) -> CIMType:
        try:
            return self._obj.getDefinition('V3')  # type: ignore (Something weird here)
        except json.JSONDecodeError:
            print('Invalid layer definition found')
            return CIMDefinition(name='INVALID CIM')  # pyright: ignore[reportReturnType]

    @property
    def unique_name(self) -> str:
        """Get the longName or name of the object. Use id for any object without a name attribute"""
        return getattr(self._obj, 'longName', None) or getattr(self._obj, 'name', None) or str(id(self._obj))

    @property
    def uri(self) -> str:
        """Get the URI for the object or the id with `:NO_URI` at the end"""
        uri = getattr(self._obj, 'URI', None)
        if uri is None:
            uri = json.loads(self._obj._arc_object.GetCimJSONString())['uRI']  # type: ignore
        return uri

    @property
    def cim_dict(self) -> dict[str, Any] | None:
        if _cim := self.cim:
            return json.loads(json.dumps(_cim, cls=CimJsonEncoder, indent=2))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._obj, attr)

    def __repr__(self) -> str:
        return f"{self._obj.__class__.__name__}({name_of(self, skip_uri=True)})"

    def __eq__(self, other: MappingWrapper[Any, Any] | Any) -> bool:
        if hasattr(other, '_obj'):
            return self._obj is getattr(other, '_obj', None)
        else:
            return super().__eq__(other)


# Wrappers around existing Mapping Objects to allow extensible functionality
class Layer(MappingWrapper[_Layer, CIMBaseLayer], _Layer):
    """mp.Layer wrapper"""
    symbology: Symbology

    @property
    def feature_class(self) -> FeatureClass:
        """Get a `arcpie.FeatureClass` object that is initialized using the layer and its current state"""
        return FeatureClass.from_layer(self)

    @property
    def lyrx(self) -> dict[str, Any] | None:
        """Get a dictionary representation of the layer that can be saved to an lyrx file using `json.dumps`

        Note:
            GroupLayer objects will return a lyrx template with all sub-layers included
        """
        cim_def = self.cim_dict
        if cim_def is None:
            return None
        lyrx: dict[str, Any] = {  # Base required keys for lyrx file
            'type': 'CIMLayerDocument',
            'layers': [self.uri],
            'layerDefinitions': [cim_def],
        }

        # Handle Group Layers
        if self.isGroupLayer and self.parent and isinstance(self.parent, Map):
            if child_tables := list(cim_def.get('tables', [])):
                child_tables: list[str]
                table_uris = set(child_tables) & set(self.parent.tables.uris)  # Skip dead nodes
                lyrx['tableDefinitions'] = [self.parent.tables[uri].cim_dict for uri in table_uris]
            if child_layers := list(cim_def.get('layers', [])):
                child_layers: list[str]
                layer_uris = set(child_layers) & set(self.parent.layers.uris)  # Skip dead nodes
                lyrx['layerDefinitions'] = [cim_def] + [self.parent.layers[uri].cim_dict for uri in layer_uris]
        return lyrx

    @property
    def cim(self) -> CIMBaseLayer:
        return super().cim

    @property
    def cim_dict(self) -> SchemaLayer | None:  # type: ignore
        return super().cim_dict  # type: ignore

    def export_lyrx(self, out_dir: Path | str) -> None:
        """Export the layer to a lyrx file in the target directory

        Args:
            out_dir (Path|str): The location to export the mapx to
        """
        out_dir = Path(out_dir)
        target = out_dir / f'{safe_name(self.longName)}.lyrx'
        # Make Containing directory for grouped layers
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_text(json.dumps(self.lyrx, indent=2), encoding='utf-8')

    def import_lyrx(self, lyrx: Path | str) -> None:
        """Import the layer state from an lyrx file

        Args:
            lyrx (Path|str): The lyrx file to update this layer with

        Note:
            CIM changes require the APRX to be saved to take effect. If you are accessing this
            layer via a Project, use `project.save()` after importing the layerfile
        """
        lay_file = LayerFile(str(lyrx))
        lyrx_layers = {lay.name: lay for lay in lay_file.listLayers() if hasattr(lay, 'name')}
        if not (lyrx_layer := lyrx_layers.get(self.name)):
            print(f'{self.name} not found in {lyrx!s}')
        else:
            # Update Connection
            lyrx_layer = Layer(lyrx_layer)
            lyrx_cim_dict = lyrx_layer.cim_dict or {}
            lyrx_layer_cim = lyrx_layer.cim
            if lyrx_cim_dict.get('featureTable') and lyrx_cim_dict['featureTable'].get('dataConnection'):
                lyrx_layer_cim.featureTable.dataConnection = self.cim.featureTable.dataConnection  # type: ignore (this is how CIM works)
            try:
                self.setDefinition(lyrx_layer_cim)  # pyright: ignore[reportArgumentType]
            except AttributeError:
                print(f'Failed to update CIM for {self.__class__.__name__}{self.name}')


class Bookmark(MappingWrapper[_Bookmark, CIMBookmark], _Bookmark): ...


class BookmarkMapSeries(MappingWrapper[_BookmarkMapSeries, CIMBookmarkMapSeries], _BookmarkMapSeries):
    """Wrapper around an arcpy.mp BookmarkMapSeries object that provides an ergonomic interface"""

    def __iter__(self) -> Iterator[BookmarkMapSeries]:
        orig_page = self.currentPageNumber
        for page in range(1, self.pageCount):
            self.currentPageNumber = page
            yield self
        if orig_page:
            self.currentPageNumber = orig_page

    def __getitem__(self, page: int | str | _Bookmark) -> BookmarkMapSeries:
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


class MapSeries(MappingWrapper[_MapSeries, CIMMapSeries], _MapSeries):
    """Wrapper around an arcpy.mp MapSeries object that provides an ergonomic interface"""
    @property
    def layer(self) -> Layer:
        """Get the mapseries target layer"""
        return Layer(self.indexLayer, self.map)

    @property  # Passthrough
    def feature_class(self) -> FeatureClass:
        """Get the FeatureClass of the parent layer"""
        return self.layer.feature_class

    @property
    def map(self) -> Map:
        """Get the map object that is being seriesed"""
        return Map(self.mapFrame.map, self.parent.parent)  # type: ignore (if a MapSeries is initialized this will be a map)

    @property
    def pageRow(self):  # type: ignore (prevent access to uninitialized pageRow raising RuntimeError)
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
            return {}  # pageRow is unset with no active page

        return {f: getattr(self.pageRow, f, self.pageRow.get(f)) for f in self.page_field_names}

    @property
    def current_page_name(self) -> str:
        """Get the name of the active mapseries page"""
        return self.page_values.get(self.page_field, 'No Page')

    def to_pdf(self, **settings: Unpack[MapSeriesPDFSetting]) -> BytesIO:
        from arcpy.mp import CreateExportFormat, CreateExportOptions  # type: ignore  # noqa: PLC0415
        if TYPE_CHECKING:
            from arcpy.mp import MapSeriesExportOptions, PDFFormat  # noqa: PLC0415
        else:
            PDFFormat = MapSeriesExportOptions = object

        with NamedTemporaryFile() as tmp:
            settings = MapseriesPDFDefault.copy()
            settings.update(settings)
            pdf: PDFFormat = CreateExportFormat('PDF', tmp.name)  # type: ignore
            pdf.resolution = settings.get('resolution', 96)
            pdf.imageQuality = settings.get('image_quality', 'BEST')
            pdf.compressVectorGraphics = settings.get('compress_vector_graphics', True)
            pdf.imageCompression = settings.get('image_compression', 'ADAPTIVE')
            pdf.embedFonts = settings.get('embed_fonts', True)
            pdf.layersAndAttributes = settings.get('layers_attributes', 'LAYERS_ONLY')
            pdf.georefInfo = settings.get('georef_info', True)
            pdf.imageCompressionQuality = settings.get('jpeg_compression_quality', 80)
            pdf.clipToElements = settings.get('clip_to_elements', False)
            pdf.outputAsImage = settings.get('output_as_image', False)
            pdf.embedColorProfile = settings.get('embed_color_profile', True)
            pdf.includeAccessibilityTags = settings.get('pdf_accessibility', False)
            pdf.removeLayoutBackground = not settings.get('keep_layout_background', True)
            pdf.convertMarkers = settings.get('convert_markers', False)
            pdf.simulateOverprint = settings.get('simulate_overprint', False)

            ms_opts: MapSeriesExportOptions = CreateExportOptions('MAPSERIES')  # type: ignore
            if 'page_range_string' in settings:
                ms_opts.customPages = settings['page_range_string']

            if 'multiple_files' in settings:
                mf = settings.get('multiple_files', 'PDF_SINGLE_FILE')
                mf = mf if mf != 'PDF_MULTIPLE_FILES_PAGE_NAME' else 'MULTIPLE_FILES_PAGE_NAME'
                mf = mf if mf != 'PDF_MULTIPLE_FILES_PAGE_NUMBER' else 'MULTIPLE_FILES_PAGE_NUMBER'
                ms_opts.setExportFileOptions(mf)

            t = settings.get('page_range_type', 'ALL')
            if t == 'RANGE':
                t = 'CUSTOM'
            elif t == 'SELECTED':
                t = 'SELECTED_INDEX_FEATURES'
            ms_opts.setExportPages(t)
            ms_opts.showExportCount = settings.get('show_export_count', False)

            with Path(self.export(pdf, ms_opts)).open('rb') as fl:
                return BytesIO(fl.read())

    def _to_pdf(self, **settings: Unpack[MapSeriesPDFSetting]) -> BytesIO:
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
        settings = MapseriesPDFDefault.copy()
        settings.update(settings)
        with NamedTemporaryFile() as tmp:
            pdf = self.exportToPDF(tmp.name, **settings)
            with Path(pdf).open('rb') as fl:
                return BytesIO(fl.read())

    def __iter__(self) -> Iterator[MapSeries]:
        orig_page = self.currentPageNumber
        for page in range(1, self.pageCount + 1):
            self.currentPageNumber = page
            yield self
        if orig_page:
            self.currentPageNumber = orig_page

    def __getitem__(self, page: int | str) -> MapSeries:
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
        return f'MapSeries<{self.layer.unique_name} @ {self.current_page_name}>'


class ElevationSurface(MappingWrapper[_ElevationSurface, CIMElevationSurfaceLayer], _ElevationSurface): ...


class Map(MappingWrapper[_Map, CIMMapDocument], _Map):
    @cached_property
    def layers(self) -> LayerManager:
        """Get a LayerManager for all layers in the Map"""
        return LayerManager(Layer(lay, self) for lay in self.listLayers() or [])

    @cached_property
    def local_layers(self) -> LayerManager:
        """Get a LayerManager for all non-web layers in the Map"""
        return LayerManager(Layer(lay, self) for lay in self.listLayers() or [] if lay.isFeatureLayer and not lay.isWebLayer)

    @cached_property
    def tables(self) -> TableManager:
        """Get a TableManager for all tables in the Map"""
        return TableManager(Table(t, self) for t in self.listTables() or [])

    @cached_property
    def bookmarks(self) -> BookmarkManager:
        """Get a BookmarkManager for all bookmarks in the Map"""
        return BookmarkManager(Bookmark(b, self) for b in self.listBookmarks() or [])

    @cached_property
    def elevation_surfaces(self) -> ElevationSurfaceManager:
        """Get an ElevationSurfaceManager for all elevation surfaces in the Map"""
        return ElevationSurfaceManager(ElevationSurface(es, self) for es in self.listElevationSurfaces() or [])

    @property
    def mapx(self) -> dict[str, Any]:
        with NamedTemporaryFile(suffix='.mapx') as tmp:
            self.exportToMAPX(tmp.name)
            return json.loads(Path(tmp.name).read_text(encoding='utf-8'))

    @property
    def cim(self) -> CIMMapDocument:
        return super().cim

    @property
    def cim_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.cim, cls=CimJsonEncoder, indent=2))

    @overload
    def __getitem__(self, name: str) -> Layer | Table: ...
    @overload
    def __getitem__(self, name: Wildcard) -> list[Layer] | list[Table]: ...
    def __getitem__(self, name: str | Wildcard) -> Any:
        obj = self.layers.get(name, None) or self.tables.get(name, None)
        if obj is None:
            raise KeyError(f'{name} not found in map {self.unique_name}')
        return obj

    @overload
    def get(self, name: str, default: _Default) -> Layer | Table | _Default: ...
    @overload
    def get(self, name: Wildcard, default: _Default) -> list[Layer] | list[Table] | _Default: ...
    def get(self, name: str | Wildcard, default: _Default = None) -> Any | _Default:
        try:
            return self[name]
        except KeyError:
            return default

    def export_mapx(self, out_dir: Path | str) -> None:
        """Export the map definitions to a mapx file in the target directory

        Args:
            out_dir (Path|str): The location to export the mapx to
        """
        target = Path(out_dir) / f'{self.unique_name}'
        target.write_text(json.dumps(self.mapx, indent=2), encoding='utf-8')

    def export_assoc_lyrx(self, out_dir: Path | str, *, skip_groups: bool = False, skip_grouped: bool = False) -> None:
        """Export all child layers to lyrx files the target directory

        Args:
            out_dir (Path|str): The location to export the lyrx files to
            skip_groups (bool): Skip group layerfiles and export each layer individually in a group subdirectory (default: False)
            skip_grouped (bool): Inverse of skip groups and instead only exports the group lyrx, skipping the individual layers (default: False)
        """
        out_dir = Path(out_dir)
        for layer in self.layers:
            if layer.isGroupLayer and skip_groups:
                continue
            try:
                layer.export_lyrx(out_dir / self.unique_name)
            except json.JSONDecodeError as e:
                print(f'Failed to export layer: {layer}: {e}')

        for table in self.tables:
            try:
                table.export_lyrx(out_dir / self.unique_name)
            except json.JSONDecodeError as e:
                print(f'Failed to export table: {table}: {e}')

    def import_assoc_lyrx(self, lyrx_dir: Path | str, *, skip_groups: bool = False) -> None:
        """Imports lyrx files that were exported using the `export_assoc_lyrx` method

        Args:
            lyrx_dir (Path|str): The directory containing the previously exported lyrx files

        Note:
            CIM changes require the APRX to be saved to take effect. If you are accessing this
            layer via a Project, use `project.save()` after importing the layerfile
        """
        lyrx_dir = Path(lyrx_dir)
        for lyrx_path in lyrx_dir.rglob('*.lyrx'):
            lyrx_name = str(lyrx_path.relative_to(lyrx_dir).with_suffix(''))
            # Since layers can have invalid file names, we need to check the `safe` name as well
            safe_layers = {safe_name(n): n for n in self.layers.names}
            safe_tables = {safe_name(n): n for n in self.tables.names}
            if lyrx_name in self.layers or lyrx_name in safe_layers:
                lyrx_name = safe_layers.get(lyrx_name, lyrx_name)
                if self.layers[lyrx_name].isGroupLayer and skip_groups:
                    continue
                self.layers[lyrx_name].import_lyrx(lyrx_path)
            elif lyrx_name in self.tables or lyrx_name in safe_tables:
                lyrx_name = safe_tables.get(lyrx_name, lyrx_name)
                self.tables[lyrx_name].import_lyrx(lyrx_path)

    def import_mapx(self, mapx: Path | str) -> None:
        raise NotImplementedError


class Layout(MappingWrapper[_Layout, CIMLayout], _Layout):

    @property
    def mapseries(self) -> MapSeries | BookmarkMapSeries | None:
        """Get the Layout MapSeries/BookmarkMapSeries if it exists"""
        if not self.mapSeries:
            return None
        if isinstance(self.mapSeries, _MapSeries):
            return MapSeries(self.mapSeries, self)
        return BookmarkMapSeries(self.mapSeries, self)

    @property
    def pagx(self) -> dict[str, Any]:
        """Access the raw CIM dictionary of the layout

        Returns:
            (dict[str, Any]): A dictionary representation of the pagx json
        """
        with NamedTemporaryFile() as tmp:
            self.exportToPAGX(tmp.name)
            return json.loads(Path(f'{tmp.name}.pagx').read_text(encoding='utf-8'))

    def to_pdf(self, **settings: Unpack[PDFSetting]) -> BytesIO:
        from arcpy.mp import CreateExportFormat  # type: ignore  # noqa: PLC0415
        if TYPE_CHECKING:
            from arcpy.mp import PDFFormat  # noqa: PLC0415
        else:
            PDFFormat = object

        with NamedTemporaryFile() as tmp:
            pdf: PDFFormat = CreateExportFormat('PDF', tmp.name)  # type: ignore
            settings = PDFDefault.copy()
            for arg in settings:
                if val := settings.get(arg):
                    settings[arg] = val
            pdf.resolution = settings.get('resolution', 96)
            pdf.imageQuality = settings.get('image_quality', 'BEST')
            pdf.compressVectorGraphics = settings.get('compress_vector_graphics', True)
            pdf.imageCompression = settings.get('image_compression', 'ADAPTIVE')
            pdf.embedFonts = settings.get('embed_fonts', True)
            pdf.layersAndAttributes = settings.get('layers_attributes', 'LAYERS_ONLY')
            pdf.georefInfo = settings.get('georef_info', True)
            pdf.imageCompressionQuality = settings.get('jpeg_compression_quality', 80)
            pdf.clipToElements = settings.get('clip_to_elements', False)
            pdf.outputAsImage = settings.get('output_as_image', False)
            pdf.embedColorProfile = settings.get('embed_color_profile', True)
            pdf.includeAccessibilityTags = settings.get('pdf_accessibility', False)
            pdf.removeLayoutBackground = not settings.get('keep_layout_background', True)
            pdf.convertMarkers = settings.get('convert_markers', False)
            pdf.simulateOverprint = settings.get('simulate_overprint', False)

            # NEW
            pdf.showSelectionSymbology = settings.get('show_selection_symbology', False)

            with Path(self.export(pdf)).open('rb') as fl:
                return BytesIO(fl.read())

    def _to_pdf(self, **settings: Unpack[PDFSetting]) -> BytesIO:
        """Get the bytes for a pdf export of the Layout

        Args:
            **settings (PDFSetting): Optional settings for the export (default: `PDFDefault`)

        Returns:
            (bytes): Raw bytes of the PDF for use in a write operation or stream

        Example:
            ```python
                # Get the layout object from a project file
                lyt = prj.layouts['Layout_1']
                # Create a pdf then write the output of to_pdf to it
                pdf = Path('pdf_path').write_bytes(lyt.to_pdf())
            ```
        """
        with NamedTemporaryFile() as tmp:
            settings = PDFDefault.copy()
            for arg in settings:
                if val := settings.get(arg):
                    settings[arg] = val
            pdf = self.exportToPDF(tmp.name, **settings)
            with Path(pdf).open('rb') as fl:
                return BytesIO(fl.read())


class Table(MappingWrapper[_Table, CIMStandaloneTable], _Table):
    @property
    def table(self) -> DataTable:
        """Get an `arcpie.Table` object from the TableLayer"""
        return DataTable.from_table(self)

    @property
    def cim(self) -> CIMStandaloneTable:
        return super().cim  # pyright: ignore[reportReturnType]

    @property
    def lyrx(self) -> dict[str, Any]:
        cim_def = self.cim_dict
        lyrx: dict[str, Any] = {  # Base required keys for lyrx file
            'type': 'CIMLayerDocument',
            'tables': [self.uri],
            'tableDefinitions': [cim_def],
        }
        return lyrx

    def export_lyrx(self, out_dir: Path | str) -> None:
        """Export the layer to a lyrx file in the target directory

        Args:
            out_dir (Path|str): The location to export the lyrx to
        """
        target = Path(out_dir) / f'{safe_name(self.longName)}.lyrx'
        # Make Containing directory for grouped layers
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_text(json.dumps(self.lyrx, indent=2), encoding='utf-8')

    def import_lyrx(self, lyrx: Path | str) -> None:
        """Import the table state from an lyrx file

        Args:
            lyrx (Path|str): The lyrx file to update this table with

        Note:
            CIM changes require the APRX to be saved to take effect. If you are accessing this
            layer via a Project, use `project.save()` after importing the layerfile
        """
        lay_file = LayerFile(str(lyrx))
        lyrx_layers = {t.longName: t for t in lay_file.listTables()}
        for table in [self]:
            lyrx_table = lyrx_layers.get(table.longName)
            if not lyrx_table:
                print(f'{self.longName} not found in {lyrx!s}')
                continue
            # Update Connection
            lyrx_table.updateConnectionProperties(None, table.connectionProperties)  # type: ignore
            lyrx_layer_cim = lyrx_table.getDefinition('V3')
            self.setDefinition(lyrx_layer_cim)


class Report(MappingWrapper[_Report, CIMReport], _Report): ...


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


def name_of(o: MappingWrapper[Any, Any], skip_uri: bool = False, uri_only: bool = False) -> str:
    """Handle the naming hierarchy of mapping objects URI -> longName -> name

    Allow setting flags to get specific names

    Note:
        If a URI is requested and no `URI` attribute is available in object,
         `'obj.name: NO URI(id(obj))'` will be returned, e.g. `'my_bookmark: NO URI(1239012093)'`
    """
    uri: str | None = o.uri if not skip_uri else None
    long_name: str | None = getattr(o, 'longName', None)  # longName will identify Grouped Layers
    name: str | None = getattr(o, 'name', None)
    oid: str = str(id(o))  # Fallback to a locally unique id (should never happen)
    if uri_only:
        return uri or f"{id(o)}:NO_URI"
    return uri or long_name or name or oid


class Manager[MappingObject: (Layer, Bookmark, BookmarkMapSeries, MapSeries, ElevationSurface, Map, Layout, Table, Report)]:
    """Base access interfaces for all manager classes. Specific interfaces are defined in the subclass

    Index itentifiers are URI -> longName -> name depending on what is available in the managed class
    """

    def __init__(self, objs: Iterable[MappingObject]) -> None:
        self._objects: dict[str, MappingObject] = {}
        for o in objs:
            if (_uri := o.uri) not in self._objects:
                self._objects[_uri] = o

    @property
    def objects(self) -> list[MappingObject]:
        """Get a list of all managed objects"""
        return list(self._objects.values())

    @property
    def names(self) -> list[str]:
        """Get the names of all managed objects (skips URIs)"""
        return [o.unique_name for o in self.objects]

    @property
    def uris(self) -> list[str]:
        """Get URIs/CIMPATH for all managed objects

        Note:
            Default to a Python id() call if no URI is present
        """
        return [o.uri for o in self.objects]

    @overload
    def __getitem__(self, name: str) -> MappingObject: ...
    @overload
    def __getitem__(self, name: int) -> MappingObject: ...
    @overload
    def __getitem__(self, name: Wildcard) -> list[MappingObject]: ...
    @overload
    def __getitem__(self, name: re.Pattern[str]) -> list[MappingObject]: ...
    @overload
    def __getitem__(self, name: slice) -> list[MappingObject]: ...

    def __getitem__(self, name: str | Wildcard | re.Pattern[str] | int | slice) -> MappingObject | list[MappingObject]:
        """Access objects using a regex pattern (re.compile), wildcard (*STRING*), index, slice, name, or URI"""
        if isinstance(name, str) and '*' in name:
            name = Wildcard(name)  # Allow passing a wildcard directly (lose type inference)
        match name:
            case int() | slice():
                return self.objects[name]  # Will raise IndexError
            case str(name) if name in self._objects:
                return self._objects[name]
            case str(name) if name in self.names:
                for o in self.objects:
                    if o.unique_name == name:
                        return o
            case re.Pattern():
                return [o for o in self.objects if name.match(name_of(o, skip_uri=True))]
            case Wildcard():
                return [o for o in self.objects if all(part in name_of(o, skip_uri=True) for part in name.split('*'))]
            case _:
                pass  # Fallthrough to raise a KeyError

        raise KeyError(f'{name} not found in objects: ({self.names})')

    @overload
    def get(self, name: str) -> MappingObject: ...
    @overload
    def get(self, name: Wildcard) -> list[MappingObject]: ...
    @overload
    def get(self, name: str, default: _Default) -> MappingObject | _Default: ...
    @overload
    def get(self, name: Wildcard, default: _Default) -> list[MappingObject] | _Default: ...

    def get(self, name: str | Wildcard, default: _Default | None = None) -> MappingObject | list[MappingObject] | _Default | None:
        """Get a value from the Project with a safe default value"""
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name: str | MappingObject) -> bool:
        """Check to see if a URI/name is present in the Manager"""
        match name:
            case str():
                return True if self.get(name) else False
            case _:
                return any(o == name for o in self.objects)

    def __iter__(self) -> Iterator[MappingObject]:
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
    def __init__(self, aprx_path: str | Path | Literal['CURRENT'] = 'CURRENT') -> None:
        self._path = str(aprx_path)

    def __repr__(self) -> str:
        return f"Project({Path(self.aprx.filePath).stem}.aprx)"

    def start(self) -> None:
        import os  # noqa: PLC0415
        os.startfile(self.aprx.filePath)  # noqa: S606

    @property
    def name(self) -> str:
        """Get the file name of the wrapped aprx minus the file extension"""
        return Path(self.aprx.filePath).stem

    @property
    def aprx(self) -> ArcGISProject:
        """Get the base ArcGISProject for the Project"""
        return ArcGISProject(self._path)

    @cached_property
    def maps(self) -> MapManager:
        """Get a MapManager for the Project maps"""
        return MapManager(Map(m, self) for m in self.aprx.listMaps())

    @cached_property
    def layouts(self) -> LayoutManager:
        """Get a LayoutManager for the Project layouts"""
        return LayoutManager(Layout(lay, self) for lay in self.aprx.listLayouts())

    @cached_property
    def reports(self) -> ReportManager:
        """Get a ReportManager for the Project reports"""
        return ReportManager(Report(r, self) for r in self.aprx.listReports())

    @property
    def broken_layers(self) -> LayerManager:
        """Get a LayerManager for all layers in the project with broken datasources"""
        return LayerManager(Layer(lay, self) for lay in self.aprx.listBrokenDataSources() if isinstance(lay, _Layer))

    @property
    def broken_tables(self) -> TableManager:
        """Get a TableManager for all tables in the project with broken datasources"""
        return TableManager(Table(t, self) for t in self.aprx.listBrokenDataSources() if isinstance(t, _Table))

    @property
    def tree(self) -> dict[str, Any]:
        return {
            repr(self):
                {
                    'maps': {
                        repr(m):
                            {
                                'tables': m.tables.names,
                                'layers': m.layers.names,
                            }
                        for m in self.maps
                    },
                    'layouts': self.layouts.names,
                    'reports': self.reports.names,
                    'broken': {
                        'layers': self.broken_layers.names,
                        'tables': self.broken_layers.names,
                    }
                }
        }

    @overload
    def __getitem__(self, name: str) -> Map | Layout | Report: ...
    @overload
    def __getitem__(self, name: Wildcard) -> list[Map] | list[Layout] | list[Report]: ...
    def __getitem__(self, name: str | Wildcard) -> Any | list[Any]:
        """Resolve the name by looking in Maps, then Layouts, then Reports"""
        obj = self.maps.get(name, None) or self.layouts.get(name, None) or self.reports.get(name, None)
        if obj is None:
            raise KeyError(f'{name} not found in {self.name}')
        return obj

    @overload
    def get(self, name: str, default: _Default) -> Map | Layout | Report | _Default: ...
    @overload
    def get(self, name: Wildcard, default: _Default) -> list[Map] | list[Layout] | list[Report] | _Default: ...
    def get(self, name: str | Wildcard, default: _Default = None) -> Any | _Default:
        try:
            return self[name]
        except KeyError:
            return default

    def save(self) -> None:
        """Save this project"""
        self.aprx.save()

    def save_as(self, path: Path | str) -> Project:
        """Saves the project under a new name

        Args:
            path (Path|str): The filepath of the new aprx

        Returns:
            (Project): A Project representing the new project file

        Note:
            Saving a Project as a new project will not update this instance, and instead returns a new
            Project instance targeted at the new file
        """
        if not str(path).endswith('.aprx'):
            path = Path(str(path) + '.aprx')
        else:
            path = Path(path)
        self.aprx.saveACopy(str(path))
        return Project(path)

    def import_pagx(self, pagx: Path | str, *, reuse_existing_maps: bool = True) -> Layout:
        """Import a pagx file into this project

        Args:
            pagx (Path|str): The path to the pagx document

        Returns:
            (Layout): A Layout parented to this project
        """
        pagx = Path(pagx)
        if not pagx.suffix == '.pagx':
            raise ValueError(f'{pagx} is not a pagx file!')

        imported = self.aprx.importDocument(str(pagx), reuse_existing_maps=reuse_existing_maps)
        # Ensure that the imported document is a Layout
        if not isinstance(imported, _Layout):
            self.aprx.deleteItem(imported)
            raise ValueError(f'{pagx} is not a valid pagx file!')

        self.refresh('layouts')
        return Layout(imported, parent=self)

    def export_pagx(self, target_dir: Path | str) -> None:
        """Export all layouts to a directory"""
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
        for layout in self.layouts:
            (target_dir / f'{layout.unique_name}.pagx').write_text(json.dumps(layout.pagx, indent=2), encoding='utf-8')

    def import_mapx(self, mapx: Path | str) -> Map:
        """Import a mapx file into this project

        Args:
            mapx (Path|str): The path to the mapx document

        Returns:
            (Map): A Map parented to this project
        """
        mapx = Path(mapx)
        if not mapx.suffix == '.mapx':
            raise ValueError(f'{mapx} is not a mapx file!')

        imported = self.aprx.importDocument(str(mapx))
        # Ensure that the imported document is a Map
        if not isinstance(imported, _Map):
            self.aprx.deleteItem(imported)
            raise ValueError(f'{mapx} is not a valid mapx file!')

        self.refresh('maps')
        return Map(imported, parent=self)

    def export_mapx(self, target_dir: Path | str) -> None:
        """Export all maps to a directory"""
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
        for m in self.maps:
            (target_dir / f'{m.unique_name}.mapx').write_text(json.dumps(m.mapx, indent=2), encoding='utf-8')

    def export_layers(self, target_dir: Path | str, *, skip_groups: bool = False, skip_grouped: bool = False) -> None:
        """Export all layers in the project to a structured directory of layerfiles

        Args:
            target_dir (Path|str): The target directory to export the layerfiles to
            skip_groups (bool): Skip group layerfiles and export each layer individually in a group subdirectory (default: False)
            skip_grouped (bool): Inverse of skip groups and instead only exports the group lyrx, skipping the individual layers (default: False)
        """
        target_dir = Path(target_dir)
        for m in self.maps:
            m.export_assoc_lyrx(target_dir, skip_groups=skip_groups, skip_grouped=skip_grouped)

    def import_layers(self, src_dir: Path | str) -> None:
        """Import a structured directory of layerfiles generated with `export_layers`

        Args:
            src_dir (Path|str): A directory containing layer files in map directories and group directories

        Note:
            CIM changes require the APRX to be saved to take effect. If you are accessing this
            layer via a Project, use `project.save()` after importing the layerfile
        """
        src_dir = Path(src_dir)
        for m in self.maps:
            map_dir = src_dir / m.unique_name
            if not map_dir.exists():
                print(f'Map {m.unique_name} does not have a valid source in the source directory, skipping')
                continue
            m.import_assoc_lyrx(map_dir)

        self.refresh('layers')

    def refresh(self, *managers: str) -> None:
        """Clear cached object managers

        Args:
            *managers (*str): Optionally limit cache clearing to certain managers (attribute name)
        """
        for prop in list(self.__dict__):
            if prop.startswith('_') or (managers and prop not in managers):
                continue  # Skip private instance attributes and non-requested
            self.__dict__.pop(prop, None)
