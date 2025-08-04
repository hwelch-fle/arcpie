from typing import TypedDict, NamedTuple, Literal, Any


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

class EsriJsonFeature(TypedDict, total=False)
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