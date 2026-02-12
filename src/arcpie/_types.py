"""Private module with type overrides for arcpy objects that make usage easier"""

from __future__ import annotations

from datetime import date, time

from collections.abc import (
    Iterator, 
    Iterable,
    Sequence,
)

from pathlib import Path
from typing import (
    Literal,
    Self, 
    Any, 
    TypeVarTuple,
    TYPE_CHECKING,
    TypedDict,
    Required,
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
        case np.float64:
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

# New for 3.6 (mp.PDFFormat)
class PDFFormatSetting(TypedDict):
    clipToElements: bool
    compressVectorGraphics: bool
    convertMarkers: bool
    embedColorProfile: bool
    embedFonts: bool
    georefInfo: bool
    height: float
    imageCompression: Literal['ADAPTIVE', 'DEFLATE', 'JPEG', 'JPEG2000', 'LZW', 'NONE', 'RLE']
    imageCompressionQuality: int
    imageQuality: Literal['BEST', 'BETTER', 'NORMAL', 'FASTER', 'FASTEST']
    includeAccessibilityTags: bool
    includeNonVisibleMapLayers: bool
    layersAndAttributes: Literal['LAYERS_AND_ATTRIBUTES', 'LAYERS_ONLY', 'NONE']
    outputAsImage: bool
    rasterAsSingleTile: bool
    removeLayoutBackground: bool
    resolution: int
    showSelectionSymbology: bool
    simulateOverprint: bool
    width: float

DefaultPDFExport: PDFFormatSetting = {
    'clipToElements': False,
    'compressVectorGraphics': True,
    'convertMarkers': False,
    'embedColorProfile': True,
    'embedFonts': True,
    'georefInfo': True,
    'height': 960,
    'imageCompression': 'ADAPTIVE',
    'imageCompressionQuality': 80,
    'imageQuality': 'NORMAL',
    'includeAccessibilityTags': True,
    'includeNonVisibleMapLayers': False,
    'layersAndAttributes': 'LAYERS_ONLY',
    'outputAsImage': False,
    'rasterAsSingleTile': False,
    'removeLayoutBackground': False,
    'resolution': 96,
    'showSelectionSymbology': False,
    'simulateOverprint': False,
    'width': 960
}

# Utility functions to make conversion between case standards easier
def snake_to_camel(s: str) -> str:
    words = s.split('_')
    words = [words[0]] + list(map(str.title, words[1:]))
    return ''.join(words)

_camel_map = {
    'A': '_a',
    'B': '_b',
    'C': '_c',
    'D': '_d',
    'E': '_e',
    'F': '_f',
    'G': '_g',
    'H': '_h',
    'I': '_i',
    'J': '_j',
    'K': '_k',
    'L': '_l',
    'M': '_m',
    'N': '_n',
    'O': '_o',
    'P': '_p',
    'Q': '_q',
    'R': '_r',
    'S': '_s',
    'T': '_t',
    'U': '_u',
    'V': '_v',
    'W': '_w',
    'X': '_x',
    'Y': '_y',
    'Z': '_z'
}

def camel_to_snake(s: str) -> str:
    return ''.join([_camel_map.get(c, c) for c in s])

def get_pdf_format(pdf_path: str|Path, setting: PDFFormatSetting):
    from arcpy.mp import PDFFormat
    fmt = PDFFormat(str(pdf_path))
    for k, v in setting.items():
        if not hasattr(fmt, k):
            # allow using snake or camel (simulate_overprint == simulateOverprint)
            k = snake_to_camel(k)
        setattr(fmt, k, v)
    return fmt

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

# System representation of an attribute rule
class AttributeRule(TypedDict):
        id: int
        name: str
        type: _CalculationType
        evaluationOrder: int
        fieldName: str
        subtypeCode: int
        description: str
        errorNumber: int
        errorMessage: str
        userEditable: bool
        isEnabled: bool
        referencesExternalService: bool
        excludeFromClientEvaluation: bool
        scriptExpression: str
        triggeringEvents: list[_TriggerEvent]
        checkParameters: dict[str, Any]
        category: int
        severity: int
        tags: str
        batch: bool
        requiredGeodatabaseClientVersion: str
        creationTime: int
        subtypeCodes: list[int]
        triggeringFields: list[str]

def convert_rule(rule: AttributeRule) -> dict[str, Any]:
    """Convert system keys to Python keys"""
    attr_map = {
        'evaluationOrder': 'evaluation_order',
        'fieldName': 'field',
        'subtypeCode': 'subtype',
        'errorNumber': 'error_number',
        'errorMessage': 'error_message',
        'userEditable': 'is_editable',
        'isEnabled': 'enabled',
        'excludeFromClientEvaluation': 'exclude_from_client_evaluation',
        'scriptExpression': 'script_expression',
        'triggeringEvents': 'triggering_events',
        'triggeringFields': 'triggering_fields',
    }
    _converted: dict[str, Any] = {}
    _converted['triggering_events'] = [
        e.removeprefix('esriARTE').upper()
        for e in rule['triggeringEvents']
    ]
    _converted['type'] = rule['type'].removeprefix('esriART').upper()
    _converted['exclude_from_client_evaluation'] = (
        'EXCLUDE' 
        if rule['excludeFromClientEvaluation'] 
        else 'INCLUDE'
    )
    _converted['batch'] = (
        'BATCH' 
        if rule['batch'] 
        else 'NOT_BATCH'
    )
    _converted['is_editable'] = (
        'EDITABLE' 
        if rule['userEditable'] 
        else 'NONEDITABLE'
    )
    _converted['subtype'] = [s for s in rule['subtypeCodes']]
    for k in rule:
        _conv_key = attr_map.get(k, k)
        # -1 is a flag for None that needs to be converted
        if isinstance(rule[k], int) and rule[k] < 0:
            rule[k] = None
        
        # Skip manually converted values
        if _conv_key in _converted:
            continue
        
        _converted[_conv_key] = rule[k]
    
    return _converted

def to_rule_alter(rule: AttributeRule) -> AlterRuleOpts:
    """Convert a system AttributeRule to a set of key value pairs that can be used with AlterAttributeRule"""
    _rule = convert_rule(rule) # Will always have correct keys
    _keys = {
        *AlterRuleOpts.__optional_keys__,
        *AlterRuleOpts.__required_keys__
    }
    return AlterRuleOpts(**{k: v for k, v in _rule.items() if k in _keys})  # pyright: ignore[reportArgumentType]

def to_rule_add(rule: AttributeRule) -> AddRuleOpts:
    """Convert a system AttributeRule to a set of key value pairs that can be used with AddAttributeRule"""
    _rule = convert_rule(rule) # Will always have correct keys
    _keys = {
        *AddRuleOpts.__optional_keys__,
        *AddRuleOpts.__required_keys__
    } # Linter does not understand this _keys check
    return AddRuleOpts(**{k: v for k, v in _rule.items() if k in _keys})  # pyright: ignore[reportArgumentType]

# Typed passthough options for Attribute Rule functions
class AlterRuleOpts(TypedDict, total=False):
    """Typed kwargs for AlterAttributeRule"""
    name: Required[str]
    script_expression: str
    triggering_events: list[Literal['INSERT', 'DELETE', 'UPDATE']]
    error_number: int
    error_message: str
    description: str | Literal['RESET']
    exclude_from_client_evaluation: Literal['EXCLUDE', 'INCLUDE']
    tags: str
    triggering_fields: Sequence[str]

class AddRuleOpts(TypedDict, total=False):
    """Typed kwargs for AddAttributeRule"""
    name: Required[str]
    type: Literal['CALCULATION', 'CONSTRAINT', 'VALIDATION']
    script_expression: str
    is_editable: Literal['EDITABLE', 'NONEDITABLE']
    triggering_events: list[Literal['INSERT', 'DELETE', 'UPDATE']]
    error_number: int
    error_message: str
    description: str
    subtype: str | list[str]
    field: str
    exclude_from_client_evaluation: Literal['EXCLUDE', 'INCLUDE']
    batch: Literal['BATCH', 'NOT_BATCH']
    severity: int
    tags: str
    triggering_fields: Sequence[str]

type Description = str
type NumericType = int | float
type DateType = datetime | date | time
type CodeType = str | NumericType | DateType
_DomainType = Literal["CodedValue", "Range"]
_DomainFieldType = Literal["Short", "Long", "BigInteger", "Float", "Double", "Text", "Date", "DateOnly", "TimeOnly"]
_DomainMergePolicy = Literal["AreaWeighted", "DefaultValue", "SumValues"]
_DomainSplitPolicy = Literal["DefaultValue", "Duplicate", "GeometryRatio"]
class SystemDomain(TypedDict):
    codedValues: dict[CodeType, Description] | None
    description: Description
    domainType: _DomainType
    mergePolicy: _DomainMergePolicy
    name: str
    owner: str
    range: tuple[CodeType, CodeType] | None
    splitPolicy: _DomainSplitPolicy
    type: _DomainFieldType

class RelationshipAddRuleOpts(TypedDict, total=False):
    in_rel_class: str
    origin_subtype: str
    origin_minimum: int
    origin_maximum: int
    destination_subtype: str
    destination_minimum: int
    destination_maximum: int 

class RelationshipRemoveRuleOpts(TypedDict, total=False):
    in_rel_class: str
    origin_subtype: str
    destination_subtype: str
    remove_all: Literal['REMOVE', 'NOT_ALL']

class RelationshipOpts(TypedDict, total=False):
    origin_table: str
    destination_table: str
    out_relationship_class: str
    relationship_type: Literal["SIMPLE", "COMPOSITE"]
    forward_label: str
    backward_label: str
    message_direction: Literal["FORWARD", "BACKWARD", "BOTH", "NONE"]
    cardinality: Literal["ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_MANY"]
    attributed: Literal["ATTRIBUTED", "NONE"]
    origin_primary_key: str
    origin_foreign_key: str
    destination_primary_key: str
    destination_foreign_key: str

DomainFieldType = Literal['SHORT', 'LONG', 'BIGINTEGER', 'FLOAT', 'DOUBLE', 'TEXT', 'DATE', 'DATEONLY', 'TIMEONLY']
DomainSplitPolicy = Literal['DEFAULT', 'DUPLICATE', 'GEOMETRY_RATIO']
DomainMergePolicy = Literal['DEFAULT', 'SUM_VALUES', 'AREA_WEIGHTED']
DomainType = Literal['CODED', 'RANGE']
class AlterDomainOpts(TypedDict, total=False):
    """Use with arcpy.management.AlterDomain"""
    new_domain_name: str
    new_domain_description: str
    new_domain_owner: str
    split_policy: DomainSplitPolicy
    merge_policy: DomainMergePolicy

class CreateDomainOpts(TypedDict, total=False):
    """Use with arcpy.management.CreateDomain"""
    domain_name: str
    field_type: DomainFieldType
    domain_type: DomainType
    domain_description: str
    split_policy: DomainSplitPolicy
    merge_policy: DomainMergePolicy

def parse_domain(domain: Domain) -> SystemDomain:
    return SystemDomain(**domain.__dict__)

DomainParamMap = {
    'Short': 'SHORT', 
    'Long': 'LONG', 
    'BigInteger': 'BIGINTEGER', 
    'Float': 'FLOAT', 
    'Double': 'DOUBLE', 
    'Text': 'TEXT', 
    'Date': 'DATE', 
    'DateOnly': 'DATEONLY', 
    'TimeOnly': 'TIMEONLY',
    'CodedValue': 'CODED',
    'Range': 'RANGE',
    'DefaultValue': 'DEFAULT',
    'Duplicate': 'DUPLICATE',
    'GeometryRatio': 'GEOMETRY_RATIO',
    'SumValues': 'SUM_VALUES',
    'AreaWeighted': 'AREA_WEIGHTED',
}

def domain_param(param: str) -> Any:
    return DomainParamMap.get(param) or param

# Parameter Datatypes for Tool Parameters
ParameterDatatype = Literal[
    'analysis_cell_size',
    'DEAddressLocator',
    'DEArcInfoTable',
    'DECadastralFabric',
    'DECadDrawingDataset',
    'DECatalogRoot',
    'DECoverage',
    'DECoverageFeatureClasses',
    'DEDatasetType',
    'DEDbaseTable',
    'DEDiskConnection',
    'DEFeatureClass',
    'DEFeatureDataset',
    'DEFile',
    'DEFolder',
    'DEGeoDataServer',
    'DEGeodatasetType',
    'DEGeometricNetwork',
    'DEGlobeServer',
    'DEGPServer',
    'DEImageServer',
    'DELasDataset',
    'DELayer',
    'DEMapDocument',
    'DEMapServer',
    'DEMosaicDataset',
    'DENetworkDataset',
    'DEPrjFile',
    'DERasterBand',
    'DERasterCatalog',
    'DERasterDataset',
    'DERelationshipClass',
    'DERemoteDatabaseFolder',
    'DESchematicDataset',
    'DESchematicDiagram',
    'DESchematicDiagramClass',
    'DESchematicFolder',
    'DEServerConnection',
    'DEShapeFile',
    'DESpatialReferencesFolder',
    'DETable',
    'DETextfile',
    'DETin',
    'DETool',
    'DEToolbox',
    'DETopology',
    'DEType',
    'DEUtilityNetwork',
    'DEVPFCoverage',
    'DEVPFTable',
    'DEWCSCoverage',
    'DEWMSMap',
    'DEWorkspace',
    'Field',
    'GP3DADecimate',
    'GP3DTilesLayer',
    'GPArcInfoItem',
    'GPArealUnit',
    'GPBoolean',
    'GPCadastralFabricLayer',
    'GPCalculatorExpression',
    'GPCatalogLayer',
    'GPCellSizeXY',
    'GPCompositeLayer',
    'GPCoordinateSystem',
    'GPDataFile',
    'GPDate',
    'GPDiagramLayer',
    'GPDouble',
    'GPEncryptedString',
    'GPEnvelope',
    'GPEvaluationScale',
    'GPExtent',
    'GPFeatureLayer',
    'GPFeatureRecordSetLayer',
    'GPFieldInfo',
    'GPFieldMapping',
    'GPGALayer',
    'GPGASearchNeighborhood',
    'GPGAValueTable',
    'GPGraph',
    'GPGraphDataTable',
    'GPGroupLayer',
    'GPINFOExpression',
    'GPInternetTiledLayer',
    'GPKMLLayer',
    'GPLasDatasetLayer',
    'GPLayer',
    'GPLine',
    'GPLinearUnit',
    'GPLong',
    'GPMap',
    'GPMapServerLayer',
    'GPMDomain',
    'GPMosaicLayer',
    'GPNAHierarchySettings',
    'GPNALayer',
    'GPNetworkDatasetLayer',
    'GPNetworkDataSource',
    'GPOrientedImageryLayer',
    'GPPairwiseWeightsTable',
    'GPPoint',
    'GPPolygon',
    'GPRandomNumberGenerator',
    'GPRasterBuilder',
    'GPRasterCalculatorExpression',
    'GPRasterCatalogLayer',
    'GPRasterDataLayer',
    'GPRasterFormulated',
    'GPRasterLayer',
    'GPRecordSet',
    'GPRouteMeasureEventProperties',
    'GPSACellSize',
    'GPSAExtractValues',
    'GPSAFuzzyFunction	',
    'GPSAGDBEnvCompression',
    'GPSAGDBEnvPyramid',
    'GPSAGDBEnvStatistics',
    'GPSAGDBEnvTileSize',
    'GPSAHorizontalFactor',
    'GPSANeighborhood',
    'GPSARadius',
    'GPSARemap',
    'GPSASemiVariogram',
    'GPSATimeConfiguration',
    'GPSATopoFeatures',
    'GPSATransformationFunction',
    'GPSAVerticalFactor',
    'GPSAWeightedOverlayTable',
    'GPSAWeightedSum',
    'GPSceneServiceLayer',
    'GPSchematicLayer',
    'GPServer',
    'GPSpatialReference',
    'GPSQLExpression',
    'GPString',
    'GPStringHidden',
    'GPTableView',
    'GPTerrainLayer',
    'GPTimeUnit',
    'GPTinLayer',
    'GPTopologyLayer',
    'GPTrajectoryLayer',
    'GPType',
    'GPUtilityNetworkLayer',
    'GPValueTable',
    'GPVariant',
    'GPVectorLayer',
    'GPXYDomain',
    'GPZDomain',
    'Index',
    'NAClassFieldMap',
    'NetworkTravelMode',
]