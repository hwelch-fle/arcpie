from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import (
    Any,
    Literal,
    SupportsIndex,
    TypeAlias,
    TypedDict,
    Unpack,
    cast,
    overload,
)

from arcpy import (
    Filter,
    Parameter as _Parameter,
)

__all__ = ('Controls', 'Parameter', 'ParameterDatatype', 'Parameters')


ParameterDirection: TypeAlias = Literal[  # noqa: UP040
    'Input',
    'Output',
]


ParameterType: TypeAlias = Literal[  # noqa: UP040
    'Required',
    'Optional',
    'Derived',
]


ParameterDatatype: TypeAlias = Literal[  # noqa: UP040
    'analysis_cell_size', 'DEAddressLocator', 'DEArcInfoTable', 'DECadastralFabric', 'DECadDrawingDataset',
    'DECatalogRoot', 'DECoverage', 'DECoverageFeatureClasses', 'DEDatasetType', 'DEDbaseTable',
    'DEDiskConnection', 'DEFeatureClass', 'DEFeatureDataset', 'DEFile', 'DEFolder', 'DEGeoDataServer',
    'DEGeodatasetType', 'DEGeometricNetwork', 'DEGlobeServer', 'DEGPServer', 'DEImageServer', 'DELasDataset',
    'DELayer', 'DEMapDocument', 'DEMapServer', 'DEMosaicDataset', 'DENetworkDataset', 'DEPrjFile',
    'DERasterBand', 'DERasterDataset', 'DERelationshipClass', 'DERemoteDatabaseFolder', 'DESchematicDataset', 'DESchematicDiagram',
    'DESchematicDiagramClass', 'DESchematicFolder', 'DEServerConnection', 'DEShapeFile', 'DESpatialReferencesFolder', 'DETable',
    'DETextfile', 'DETin', 'DETool', 'DEToolbox', 'DETopology', 'DEType',
    'DEUtilityNetwork', 'DEVPFCoverage', 'DEVPFTable', 'DEWCSCoverage', 'DEWMSMap', 'DEWorkspace',
    'Field', 'GP3DADecimate', 'GP3DTilesLayer', 'GPArcInfoItem', 'GPArealUnit', 'GPBoolean',
    'GPCadastralFabricLayer', 'GPCalculatorExpression', 'GPCatalogLayer', 'GPCellSizeXY', 'GPCompositeLayer', 'GPCoordinateSystem',
    'GPDataFile', 'GPDate', 'GPDiagramLayer', 'GPDouble', 'GPEncryptedString', 'GPEnvelope',
    'GPEvaluationScale', 'GPExtent', 'GPFeatureLayer', 'GPFeatureRecordSetLayer', 'GPFieldInfo', 'GPFieldMapping',
    'GPGALayer', 'GPGASearchNeighborhood', 'GPGAValueTable', 'GPGraph', 'GPGraphDataTable', 'GPGroupLayer',
    'GPINFOExpression', 'GPInternetTiledLayer', 'GPKMLLayer', 'GPLasDatasetLayer', 'GPLayer', 'GPLine',
    'GPLinearUnit', 'GPLong', 'GPMap', 'GPMapServerLayer', 'GPMDomain', 'GPMosaicLayer',
    'GPNAHierarchySettings', 'GPNALayer', 'GPNetworkDatasetLayer', 'GPNetworkDataSource', 'GPOrientedImageryLayer', 'GPPairwiseWeightsTable',
    'GPPoint', 'GPPolygon', 'GPRandomNumberGenerator', 'GPRasterBuilder', 'GPRasterCalculatorExpression', 'GPRasterDataLayer',
    'GPRasterFormulated', 'GPRasterLayer', 'GPRecordSet', 'GPRouteMeasureEventProperties', 'GPSACellSize', 'GPSAExtractValues',
    'GPSAFuzzyFunction', 'GPSAGDBEnvCompression', 'GPSAGDBEnvPyramid', 'GPSAGDBEnvStatistics', 'GPSAGDBEnvTileSize', 'GPSAHorizontalFactor',
    'GPSANeighborhood', 'GPSARadius', 'GPSARemap', 'GPSASemiVariogram', 'GPSATimeConfiguration', 'GPSATopoFeatures',
    'GPSATransformationFunction', 'GPSAVerticalFactor', 'GPSAWeightedOverlayTable', 'GPSAWeightedSum', 'GPSceneServiceLayer', 'GPSchematicLayer',
    'GPSpatialReference', 'GPSQLExpression', 'GPString', 'GPStringHidden', 'GPTableView', 'GPTerrainLayer',
    'GPTimeUnit', 'GPTinLayer', 'GPTopologyLayer', 'GPTrajectoryLayer', 'GPType', 'GPUtilityNetworkLayer',
    'GPValueTable', 'GPVariant', 'GPVectorLayer', 'GPXYDomain', 'GPZDomain', 'Index',
    'NAClassFieldMap', 'NetworkTravelMode',
]


class Controls(StrEnum):
    EXTENT_NO_UNION = '{15F0D1C1-F783-49BC-8D16-619B8E92F668}'

    NUMERIC_SLIDER = '{C8C46E43-3D27-4485-9B38-A49F3AC588D9}'
    NUMERIC_WIDE = '{7A47E79C-9734-4167-9698-BFB00F43AE41}'

    COMPOSITE_SWITCH = '{BEDF969C-20D2-4C41-96DA-32408CA72BF6}'

    STRING_TEXT_BOX = '{E5456E51-0C41-4797-9EE4-5269820C6F0E}'

    MULTIVALUE_CHECKBOX = '{172840BF-D385-4F83-80E8-2AC3B79EB0E0}'
    MULTIVALUE_CHECKBOX_SELECT_ALL = '{38C34610-C7F7-11D5-A693-0008C711C8C1}'

    FEATURE_LAYER_CREATE = '{60061247-BCA8-473E-A7AF-A2026DDE1C2D}'

    VALUE_TABLE_HORIZONTAL = '{1AA9A769-D3F3-4EB0-85CB-CC07C79313C8}'
    VALUE_TABLE_NO_ADD = '{1A1CA7EC-A47A-4187-A15C-6EDBA4FE0CF7}'

    DATETIME_ONLY_DATE = '{499BF343-569C-4B2B-864B-742C602C33FE}'
    DATETIME_ONLY_TIME = '{4FA5E857-E8CC-4226-8B87-1BFA0A9876EC}'


@overload
def param_datatype(s: str | ParameterDatatype) -> ParameterDatatype: ...  # type: ignore
@overload
def param_datatype(s: Sequence[str | ParameterDatatype]) -> list[ParameterDatatype]: ...
def param_datatype(s: str | Sequence[str]) -> ParameterDatatype | list[ParameterDatatype]:
    datatypes = ParameterDatatype.__args__
    match s:
        case str(x):
            if x in datatypes:
                return cast(ParameterDatatype, s)
            raise ValueError(f'{s} not in {datatypes}')
        case Sequence():
            return [param_datatype(i) for i in s]
        case _:
            raise TypeError(f'Expected str | Sequecne[str], got {type(s)}')


@overload
def param_direction(s: str | ParameterDirection) -> ParameterDirection: ...
@overload
def param_direction(s: None) -> None: ...
def param_direction(s: str | ParameterDirection | None) -> ParameterDirection | None:
    directions = ParameterDirection.__args__
    if s in directions:
        return cast(ParameterDirection, s)
    raise ValueError(f'{s} not in {directions}')


@overload
def param_type(s: str | ParameterType) -> ParameterType: ...
@overload
def param_type(s: None) -> None: ...
def param_type(s: str | ParameterType | None) -> ParameterType | None:
    types = ParameterType.__args__
    if s in types:
        return cast(ParameterType, s)
    raise ValueError(f'{s} not in {types}')


def resolve_bases(param: type) -> list[str]:
    # Allow multiple inheritance of base datatypes
    bases = param.__mro__[:-5]
    if len(bases) > 1:
        bases = [ib for b in bases[1:] for ib in resolve_bases(b)]
    else:
        name = param.__name__
        if name == 'Parameter':
            name = param.__qualname__
        bases = [name]
    return sorted(set(bases))


class ParameterAttrs(TypedDict, total=False):
    filter: list[Any] | range
    options: list[Any] | range
    # default == value
    default: Any
    value: Any
    dependencies: list[str | Parameter]
    required: bool
    # For GPString only
    hidden: bool
    # GPFeatureLayer only
    allow_create: bool
    # GPValueTable only
    columns: dict[str, ParameterDatatype]
    filters: dict[str, list[Any]]
    # defaults == values
    defaults: list[dict[str, str]]
    values: list[dict[str, str]]


class Parameter(_Parameter):
    """Wrapper for arcpy.Parameter that allows multiple inheritance and post_init passthroughs"""
    def __init__(
        self,
        displayName: str,
        name: str | None = None,
        direction: ParameterDirection = 'Input',
        datatype: ParameterDatatype | Sequence[ParameterDatatype] | None = None,
        parameterType: ParameterType = 'Required',
        enabled: bool = True,
        category: str | None = None,
        symbology: str | None = None,
        multiValue: bool = False,
        **kwargs: Unpack[ParameterAttrs],
    ) -> None:

        if datatype is None:
            datatype = param_datatype(resolve_bases(type(self)))
            if len(datatype) == 1:
                datatype = datatype.pop()

        self._iscomposite = isinstance(datatype, list)

        if name is None:
            # snake case name
            name = displayName.lower().replace(' ', '_')

        # arcpy needs __class__.__name__ to be 'Parameter'
        self.__class__.__name__ = 'Parameter'
        super().__init__(
            name=name,
            displayName=displayName,
            direction=param_direction(direction) if direction else None,
            datatype=datatype,  # type: ignore
            parameterType=param_type(parameterType) if parameterType else None,
            enabled=enabled,
            category=category,
            symbology=symbology,
            multiValue=multiValue,
        )

        self.datatype: ParameterDatatype
        self.filter: Filter
        self._post_init(kwargs)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.value})'

    def _post_init(self, ctx: ParameterAttrs) -> None:
        """Runs after parameter initialization"""

        for attr, value in ctx.items():
            if attr in ('filter', 'options'):
                if isinstance(value, list):
                    self.filter.list = value
                elif isinstance(value, range):
                    if value.step == 1:
                        if not self._iscomposite:
                            self.filter.type = 'Range'
                        self.filter.list = [value.start, value.stop]
                    else:
                        self.filter.list = list(value)
            elif attr == 'default':
                self.value = value
            elif attr == 'dependencies':
                if not isinstance(value, list):
                    raise ValueError('Dependencies must be a list of parameter names or indices')
                self.parameterDependencies = value  # type: ignore
            elif attr == 'required':
                self.parameterType = 'Required' if value else 'Optional'
            else:
                setattr(self, attr, value)


class Parameters(list[Parameter]):
    """Wrap a list of parameters and override the index to allow indexing by name"""
    @overload
    def __getitem__(self, key: SupportsIndex, /) -> Parameter: ...
    @overload
    def __getitem__(self, key: slice, /) -> list[Parameter]: ...
    @overload
    def __getitem__(self, key: str, /) -> Parameter: ...
    def __getitem__(self, key: SupportsIndex | slice | str, /) -> Parameter | list[Parameter]:
        if isinstance(key, str):
            matches = [p for p in self if p.name.lower().replace(' ', '_') == key]
            if not matches:
                raise KeyError(key)
            if len(matches) == 1:
                return matches.pop()
            raise KeyError(f'{key} is used for multiple parameters')
        return self[key]

    @overload
    def get[D](self, key: SupportsIndex, default: D = None, /) -> Parameter | D: ...
    @overload
    def get[D](self, key: slice, default: D = None, /) -> list[Parameter] | D: ...
    @overload
    def get[D](self, key: str, default: D = None, /) -> Parameter | D: ...
    def get[D](self, key: SupportsIndex | slice | str, default: D = None, /) -> Parameter | list[Parameter] | D:
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
