"""Private module with type overrides for arcpy objects that make usage easier"""

from __future__ import annotations

from collections.abc import (
    Iterator, 
    Iterable,
)

from typing import (
    Literal,
    Self, 
    Any, 
    TypeVarTuple,
    TYPE_CHECKING,
    TypedDict,
    Generic,
    NamedTuple,
)

if TYPE_CHECKING:
    from arcpy import SpatialReference
    from arcpy._mp import Layer
    from arcpy.da import (
        SpatialRelationship,
        SearchOrder,
        Domain,
    )
else:
    SpatialRelationship = None
    SearchOrder = None

import numpy as np
import builtins
from datetime import datetime
from types import TracebackType
from .cursor import SQLClause

def cast_type(dt: np.dtype[Any]) -> type:
    match dt.type:
        case np.int_:
            return int
        case np.float_:
            return float
        case np.str_:
            return str
        case np.datetime64:
            return datetime
        case builtins.object:
            return builtins.object
        case _:
            return builtins.object

def convert_dtypes(dtypes: np.dtype[Any]) -> dict[str, type]:
    return {field: cast_type(dtypes[field]) for field in dtypes.names or {}}

# Typevar that can be used with a cursor to type the yielded tuples
# SearchCursor[int, str, str]('table', ['total', 'name', 'city'])
_RowTs = TypeVarTuple('_RowTs')

class SearchCursor(Iterator[tuple[*_RowTs]]):
    def __init__(
        self,
        in_table: str | Layer,
        field_names: str | Iterable[str],
        where_clause: str | None = None,
        spatial_reference: str | int | SpatialReference | None = None,
        explode_to_points: bool | None = False,
        sql_clause: SQLClause = SQLClause(None, None),
        datum_transformation: str | None = None,
        spatial_filter: Any = None,
        spatial_relationship: SpatialRelationship | None = None,
        search_order: SearchOrder | None = None,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None): ...
    def __next__(self) -> tuple[*_RowTs]: ...
    def __iter__(self) -> Iterator[tuple[*_RowTs]]: ...
    def next(self) -> tuple[*_RowTs]: ...
    def reset(self) -> None: ...
    def _as_narray(self) -> np.record: ...
    @property
    def fields(self) -> tuple[str, ...]: ...
    @property
    def _dtype(self) -> np.dtype[Any]: ...

class InsertCursor(Generic[*_RowTs]):
    _enable_simplify: bool = False
    def __init__(
        self,
        in_table: str | Layer,
        field_names: str | Iterable[str],
        datum_transformation: str | None = None,
        explicit: bool | None = False,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None): ...
    @property
    def fields(self) -> tuple[str, ...]: ...
    def insertRow(self, row: tuple[*_RowTs]) -> int: ...

class UpdateCursor(Iterator[tuple[*_RowTs]]):
    _enable_simplify: bool | None = False
    def __init__(
        self,
        in_table: str | Layer,
        field_names: str | Iterable[str],
        where_clause: str | None = None,
        spatial_reference: str | int | SpatialReference | None = None,
        explode_to_points: bool | None = False,
        sql_clause: SQLClause = SQLClause(None, None),
        skip_nulls: bool | None = False,
        null_value: Any | dict[str, Any] = None,
        datum_transformation: str | None = None,
        explicit: bool | None = False,
        spatial_filter: Any | None = None,
        spatial_relationship: SpatialRelationship | None = None,
        search_order: SearchOrder | None = None,
    ) -> None: ...
    @property
    def fields(self) -> tuple[str, ...]: ...
    def reset(self) -> None: ...
    def deleteRow(self) -> None: ...
    def updateRow(self, row: tuple[*_RowTs]) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None): ...
    def __next__(self) -> tuple[*_RowTs]: ...
    def __iter__(self) -> Iterator[tuple[*_RowTs]]: ...

class Subtype(TypedDict):
    Name: str
    FieldValues: dict[str, tuple[Any|None, Domain|None]]
    SubtypeField: str

# Esri spec from here https://developers.arcgis.com/rest/services-reference/enterprise/geometry-objects/
# TODO: Figure out the chaos that is the ESRI geometry spec

EsriJsonFieldType = Literal[
    'esriFieldTypeOID',
    'esriFieldTypeInteger',
    'esriFieldTypeString',
    'esriFieldTypeFloat',
    'esriFieldTypeDate',
    'esriFieldTypeGlobalID'
]

GeoJsonShape = Literal[
    'Point', 
    'LineString', 
    'Polygon', 
    'Multipoint', 
    'MultiLineString', 
    'MultiPolygon', 
    'GeometryCollection'
]

EsriJsonShape = Literal[
    'esriGeometryPoint',
    'esriGeometryMultipoint',
    'esriGeometryPolyline',
    'esriGeometryPolygon',
    'esriGeometryEnvelope',
]

EsriToGeoJson: dict[EsriJsonShape, GeoJsonShape] = {
    'esriGeometryPoint': 'Point',
    'esriGeometryMultipoint': 'Multipoint',
    'esriGeometryPolyline': 'LineString',
    'esriGeometryPolygon': 'Polygon',
    'esriGeometryEnvelope': 'Polygon',
}

GeoJsonToEsri: dict[GeoJsonShape, EsriJsonShape | None] = {
    'Point': 'esriGeometryPoint',
    'LineString': 'esriGeometryPolyline',
    'Polygon': 'esriGeometryPolygon',
    'Multipoint': 'esriGeometryMultipoint',
    'MultiLineString': 'esriGeometryPolyline',
    'MultiPolygon': 'esriGeometryPolygon',
    'GeometryCollection': None # No ESRI type for mixed shape collections
}

class EsriJsonField(TypedDict, total=False):
    name: str
    type: EsriJsonFieldType
    alias: str

class Point(NamedTuple):
    x: float
    y: float
    z: float
    m: float

class ComplexShapeMixin(TypedDict, total=False):
    hasZ: bool
    hasM: bool
    ids: list[list[int]]
    spatialReference: dict[str, str]

class EsriGeometryCurve(TypedDict, total=False):
    c: list[tuple[Point, tuple[float, float]]]
    a: list[tuple[Point, tuple[float, float], bool, bool]]

class EsriGeometryPoint(TypedDict, total=False):
    x: float
    y: float
    z: float
    m: float
    id: int
    spatialReference: dict[str, str]

class EsriGeometryMultipoint(ComplexShapeMixin, total=False):
    points: list[Point]


class EsriGeometryPolyline(ComplexShapeMixin, total=False):
    paths: list[Point]

class EsriGeometryPolygon(TypedDict, total=False):
    rings: list[Point]

class EsriJsonFeature(TypedDict, total=False):
    attributes: dict[str, Any]
    geometry: dict[str, Any]

class EsriJson(TypedDict, total=False):
    displayFieldName: str
    fieldAliases: dict[str, str]
    geometryType: GeoJsonShape
    hasZ: bool
    hasM: bool
    spatialReference: str
    fields: list[EsriJsonField]
    features: list[EsriJsonFeature]

class GeoJsonGeometry(TypedDict, total=False):
    type: GeoJsonShape
    coordinates: list[list[float]]
    properties: dict[str, Any]

class GeoJsonFeature(TypedDict, total=False):
    type: Literal['Feature']
    geometry: GeoJsonGeometry

class GeoJsonFeatureCollection(TypedDict, total=False):
    type: Literal['FeatureCollection']
    features: list[GeoJsonFeature]

if TYPE_CHECKING:
    from arcpy._mp import (
        ImageQuality,
        ImageCompression,
        LayerAttributes,
    )
    
class PDFSetting(TypedDict, total=False):
    resolution: int
    image_quality: ImageQuality
    compress_vector_graphics: bool
    image_compression: ImageCompression
    embed_fonts: bool
    layers_attributes: LayerAttributes
    georef_info: bool
    jpeg_compression_quality: int
    clip_to_elements: bool
    output_as_image: bool
    embed_color_profile: bool
    pdf_accessibility: bool
    keep_layout_background: bool
    convert_markers: bool
    simulate_overprint: bool

class MapSeriesPDFSetting(PDFSetting, total=False):
    page_range_type: Literal['ALL', 'CURRENT', 'RANGE', 'SELECTED']
    multiple_files: Literal['PDF_MULTIPLE_FILES_PAGE_NAME', 'PDF_MULTIPLE_FILES_PAGE_NUMBER', 'PDF_SINGLE_FILE']
    page_range_string: str

# Allow overriding this to change global export defaults for any
# class that uses this  
PDFDefault = PDFSetting(
    resolution=96,
    image_quality='BEST',
    compress_vector_graphics=True,
    image_compression='ADAPTIVE',
    embed_fonts=True,
    layers_attributes='LAYERS_ONLY',
    georef_info=True,
    jpeg_compression_quality=80,
    clip_to_elements=False,
    output_as_image= False,
    embed_color_profile=True,
    pdf_accessibility=False,
    keep_layout_background=True,
    convert_markers=False,
    simulate_overprint=False,
)

# Use general PDF defaults for MapSeries and set map series specific defaults
MapseriesPDFDefault = MapSeriesPDFSetting()
for k, v in PDFDefault.items():
    MapseriesPDFDefault[k] = v
MapseriesPDFDefault['page_range_type'] = 'ALL'
MapseriesPDFDefault['multiple_files'] = 'PDF_SINGLE_FILE'
MapseriesPDFDefault['page_range_string'] = ''

# User and System strings for Attribute Rule Events
TriggerEvent = Literal['INSERT', 'UPDATE', 'DELETE']
_TriggerEvent = Literal['esriARTEUpdate', 'esriARTEInsert', 'esriARTEDelete']

# User and System strings for Attribute Rule Types
CalculationType = Literal['CALCULATION', 'VALIDATION', 'CONSTRAINT']
_CalculationType = Literal['esriARTCalculation', 'esriARTValidation', 'esriARTConstraint']

AttributeRule = TypedDict(
    'AttributeRule',
    {
        'id': int,
        'name': str,
        'type': _CalculationType,
        'evaluationOrder': int,
        'fieldName': str,
        'subtypeCode': int,
        'description': str,
        'errorNumber': int,
        'errorMessage': str,
        'userEditable': bool,
        'isEnabled': bool,
        'referencesExternalService': bool,
        'excludeFromClientEvaluation': bool,
        'scriptExpression': str,
        'triggeringEvents': list[_TriggerEvent],
        'checkParameters': dict[str, Any],
        'category': int,
        'severity': int,
        'tags': str,
        'batch': bool,
        'requiredGeodatabaseClientVersion': str,
        'creationTime': int,
        'subtypeCodes': list[int],
        'triggeringFields': list[str],
    },
)
