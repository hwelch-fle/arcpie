from __future__ import annotations
from typing import TypedDict, Literal

from typing import Any, TypedDict, Literal

# TODO: This is a VERY rough layout of the Layer schema. It's missing a lot of Literal flags and the Renderer class 

ESRIDatasetType = Literal['esriDTFeatureClass']
ESRIWorspaceFactoryType = Literal['FileGDB']

class SchemaDataConnection(TypedDict):
    customParameters: list[Any]
    dataset: str
    datasetType: ESRIDatasetType
    featureDataset: str
    type: Literal['CIMFeatureDatasetDataConnection']
    workspaceConnectionString: str
    workspaceFactory: ESRIWorspaceFactoryType

class SchemaExpressionInfo(TypedDict):
    expression: str
    returnType: Literal['Default']
    title: str
    type: Literal['CIMExpressionInfo']

class SchemaNumberFormat(TypedDict):
    alignmentOption: Literal['esriAlignRight']
    alignmentWidth: float
    roundingOption: Literal['esriRoundNumberOfDecimals']
    roundingValue: float
    showPlusSign: bool
    type: Literal['CIMNumericFormat']
    useSeparator: bool
    zeroPad: bool

class SchemaFieldDescription(TypedDict):
    alias: str
    fieldName: str
    highlight: bool
    numberFormat: SchemaNumberFormat
    readOnly: bool
    searchMode: Literal['Exact']
    searchable: bool
    type: Literal['CIMFieldDescription']
    valueAsRatio: bool
    visible: bool

class SchemaFeatureTable(TypedDict):
    bindVariables: list[Any]
    dataConnection: SchemaDataConnection
    databaseRelates: list[str]
    definitionFilterChoices: list[Any]
    displayExpressionInfo: SchemaExpressionInfo
    editable: bool
    fieldDescriptions: list[SchemaFieldDescription]
    floorAwareTableProperties: Any
    isLicensedDataSource: bool
    rangeDefinitions: list[Any]
    relates: list[str]
    searchOrder: Literal['esriSearchOrderSpatial']
    selectRelatedData: bool
    studyArea: Any
    studyAreaSpatialRel: Literal['esriSpatialRelUndefined']
    subtypeValue: int
    timeDefinition: Any
    timeDimensionFields: list[str] | None
    timeDisplayDefinition: str | None
    timeFields: list[str] | None
    type: Literal['CIMFeatureTable']
    useSubtypeValue: bool

class SchemaFeatureTemplate(TypedDict):
    defaultValues: dict[str, Any]
    excludedToolGUIDs: list[str]
    hiddenFields: list[str]
    name: str
    relationships: list[Any]
    requiredFields: list[str]
    toolOptions: list[Any]
    type: Literal['CIMRowTemplate']

class SchemaSeparator(TypedDict):
    separator: str
    splitAfter: bool
    splitForced: bool
    type: Literal['CIMMaplexStackingSeparator']
    visible: bool

class SchemaLabelStackingProperties(TypedDict):
    maximumNumberOfCharsPerLine: int
    maximumNumberOfLines: int
    minimumNumberOfCharsPerLine: int
    preferToStackLongLabels: bool
    separators: list[SchemaSeparator]
    stackAlignment: Literal['ChooseBest']
    trimStackingSeparators: bool
    type: Literal['CIMMaplexLabelStackingProperties']

class SchemaMaplexProperties(TypedDict):
    alignLabelToLineDirection: bool
    allowAsymmetricOverrun: bool
    allowStraddleStacking: bool
    alternateLabelExpressionInfo: Any
    avoidOverlappingLabeledPolygonsAsIfHoles: bool
    avoidPolygonHoles: bool
    backgroundLabel: bool
    boundaryLabelingAllowHoles: bool
    boundaryLabelingAllowSingleSided: bool
    boundaryLabelingSingleSidedOnLine: bool
    canAbbreviateLabel: bool
    canFlipStackedStreetLabel: bool
    canKeyNumberLabel: bool
    canOverrunFeature: bool
    canPlaceLabelOnTopOfFeature: bool
    canPlaceLabelOutsidePolygon: bool
    canReduceFontSize: bool
    canReduceLeading: bool
    canRemoveOverlappingLabel: bool
    canShiftPointLabel: bool
    canStackLabel: bool
    canTruncateLabel: bool
    canUseAlternateLabelExpression: bool
    centerLabelAnchorType: Literal['Symbol']
    connectionType: Literal['Unambiguous']
    constrainOffset: Literal['NoConstraint']
    contourAlignmentType: Literal['Page']
    contourLadderType: Literal['Straight']
    contourMaximumAngle: int
    enableConnection: bool
    enablePointPlacementPriorities: bool
    enablePolygonFixedPosition: bool
    enableSecondaryOffset: bool
    featureType: Literal['Line']
    featureWeight: int
    fontHeightReductionLimit: int
    fontHeightReductionStep: float
    fontWidthReductionLimit: int
    fontWidthReductionStep: int
    graticuleAlignment: bool
    graticuleAlignmentType: Literal['Straight']
    isLabelBufferHardConstraint: bool
    isMinimumSizeBasedOnArea: bool
    isOffsetFromFeatureGeometry: bool
    keyNumberGroupName: Literal['Default']
    labelBuffer: int
    labelLargestPolygon: bool
    labelPriority: int

class SchemaLabelClass(TypedDict):
    expression: str
    expressionEngine: Literal['Arcade']
    expressionTitle: str
    featuresToLabel: Literal['AllVisibleFeatures']
    iD: int
    maplexLabelPlacementProperties: SchemaMaplexProperties

class SchemaLayer3DProperties(TypedDict):
    castShadows: bool
    depthPriority: int
    enable2DSymbolPerspectiveScaling: bool
    exaggerationMode: Literal['ScaleZ']
    isLayerLit: bool
    layerFaceCulling: Literal['None']
    lighting: Literal['OneSideDataNormal']
    maxDistance: int
    maxPreloadDistance: float
    minDistance: int
    minPreloadDistance: float
    optimizeMarkerTransparency: bool
    preloadTextureCutoffHigh: int
    preloadTextureCutoffLow: float
    textureCutoffHigh: float
    textureCutoffLow: int
    textureDownscalingFactor: int
    type: Literal['CIM3DLayerProperties']
    useCompressedTextures: bool
    useDepthWritingForTransparency: bool
    verticalExaggeration: int
    verticalUnit: dict[Literal['uwkid'], int]

class SchemaLayerElevation(TypedDict):
    isRelativeToScene: bool
    offsetZ: float
    type: Literal['CIMLayerElevationSurface']

class SchemaSourceModifiedTime(TypedDict):
    time: Any
    timeReference: Any
    type: Literal['TimeInstant']

# TODO: This is a massive class
class SchemaRenderer(TypedDict): ...

class SchemaLayer(TypedDict):
    actions: list[Any]
    allowDrapingOnIntegratedMesh: bool
    autoGenerateFeatureTemplates: bool
    blendingMode: Literal['Alpha']
    charts: list[Any]
    customProperties: list[Any]
    description: str
    displayCacheType: Literal['Permanent']
    displayFilterChoices: list[Any]
    displayFilters: list[Any]
    displayFiltersType: Literal['ByScale']
    enableDisplayFilters: bool
    enableLayerEffects: bool
    exclusionSet: list[Any]
    expanded: bool
    extrusion: Any
    featureBlendingMode: Literal['Alpha']
    featureCacheType: Literal['Session']
    featureEffects: Any
    featureElevationExpression: str
    featureElevationExpressionInfo: Any
    featureMasks: list[Any]
    featureReduction: Any
    featureSortInfos: list[Any]
    featureTable: SchemaFeatureTable
    featureTemplates: list[SchemaFeatureTemplate]
    htmlPopupEnabled: bool
    htmlPopupFormat: Any
    isFlattened: bool
    labelClasses: list[SchemaLabelClass]
    labelVisibility: bool
    layer3DProperties: SchemaLayer3DProperties
    layerEffects: list[Any]
    layerEffectsMode: Literal['Layer']
    layerElevation: SchemaLayerElevation
    layerMasks: list[Any]
    layerScaleVisibilityOptions: Any
    layerTemplate: Any
    layerType: Literal['Operational']
    maskedSymbolLayers: list[Any]
    maxDisplayCacheAge: int
    maxScale: float
    metadataURI: str
    minScale: float
    name: str
    pageDefinition: Any
    polygonSelectionFillColor: Any
    popupInfo: Any
    previousObservationsCount: int
    previousObservationsRenderer: Any
    rasterizeOnExport: bool
    refreshRate: int
    refreshRateUnit: Literal['esriTimeUnitsSeconds']
    renderer: SchemaRenderer
    scaleSymbols: bool
    searchable: bool
    selectable: bool
    selectionColor: Any
    selectionSymbol: Any
    serviceLayerID: int
    showLegends: bool
    showMapTips: bool
    showPopups: bool
    showPreviousObservations: bool
    showTracks: bool
    snappable: bool
    sourceModifiedTime: SchemaSourceModifiedTime
    symbolLayerDrawing: Any
    trackLinesRenderer: Any
    transparency: float
    type: Literal['CIMFeatureLayer']
    uRI: str
    useRealWorldSymbolSizes: bool
    useSelectionSymbol: bool
    useSourceMetadata: bool
    useVisibilityTimeExtent: bool
    visibility: bool
    visibilityTimeExtent: Any
