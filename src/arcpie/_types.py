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
    from cursor import SQLClause
    from arcpy import SpatialReference
    from arcpy._mp import Layer
    from arcpy.da import (
        SpatialRelationship,
        SearchOrder,
    )
else:
    SpatialRelationship = None
    SearchOrder = None

from numpy import (
    dtype, 
    record,
)

from types import TracebackType

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
    def _as_narray(self) -> record: ...
    @property
    def fields(self) -> tuple[str, ...]: ...
    @property
    def _dtype(self) -> dtype[Any]: ...

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