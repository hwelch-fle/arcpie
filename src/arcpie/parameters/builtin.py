from .base import Parameter

__all__ = [
    'DEAddressLocator',
    'DEArcInfoTable',
    'DECadDrawingDataset',
    'DECadastralFabric',
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
    'DEGPServer',
    'DEGeoDataServer',
    'DEGeodatasetType',
    'DEGeometricNetwork',
    'DEGlobeServer',
    'DEImageServer',
    'DELasDataset',
    'DELayer',
    'DEMapDocument',
    'DEMapServer',
    'DEMosaicDataset',
    'DENetworkDataset',
    'DEPrjFile',
    'DERasterBand',
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
    'GPMDomain',
    'GPMap',
    'GPMapServerLayer',
    'GPMosaicLayer',
    'GPNAHierarchySettings',
    'GPNALayer',
    'GPNetworkDataSource',
    'GPNetworkDatasetLayer',
    'GPOrientedImageryLayer',
    'GPPairwiseWeightsTable',
    'GPPoint',
    'GPPolygon',
    'GPRandomNumberGenerator',
    'GPRasterBuilder',
    'GPRasterCalculatorExpression',
    'GPRasterDataLayer',
    'GPRasterFormulated',
    'GPRasterLayer',
    'GPRecordSet',
    'GPRouteMeasureEventProperties',
    'GPSACellSize',
    'GPSAExtractValues',
    'GPSAFuzzyFunction',
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
    'GPSQLExpression',
    'GPSceneServiceLayer',
    'GPSchematicLayer',
    'GPSpatialReference',
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
    'analysis_cell_size',
]


class GP3DTilesLayer(Parameter):
    """A 3D tiles layer references a tile set that defines an integrated mesh or 3D object type data in a hierarchical data structure."""


class DEAddressLocator(Parameter):
    """A dataset used for geocoding that stores the address attributes, associated indexes, and rules that define the process for translating nonspatial descriptions of places to spatial data."""


class analysis_cell_size(Parameter):
    """The cell size used by raster tools."""


class GPType(Parameter):
    """A data type that accepts any value."""


class DEMapDocument(Parameter):
    """A file that contains one map, its layout, and its associated layers, tables, charts, and reports."""


class GPArealUnit(Parameter):
    """An areal unit type and value, such as square meter or acre."""


class GPBoolean(Parameter):
    """A Boolean value."""


class DECadDrawingDataset(Parameter):
    """A vector data source combined with feature types and symbology. The dataset cannot be used for feature class-based queries or analysis."""


class GPCalculatorExpression(Parameter):
    """A calculator expression."""


class GPCatalogLayer(Parameter):
    """A collection of references to different data types. The data types can be from different locations and are managed and visualized dynamically as layers based on location, time, and other attributes."""


class DECatalogRoot(Parameter):
    """The top-level node in the Catalog tree."""


class GPSACellSize(Parameter):
    """The cell size used by the ArcGIS Spatial Analyst extension."""


class GPCellSizeXY(Parameter):
    """The size that defines the two sides of a raster cell."""


class GPCompositeLayer(Parameter):
    """A reference to several children layers, including symbology and rendering properties."""


class GPSAGDBEnvCompression(Parameter):
    """The type of compression used for a raster."""


class GPCoordinateSystem(Parameter):
    """A reference framework, such as the UTM system consisting of a set of points, lines, or surfaces, and a set of rules used to define the positions of points in two- and three-dimensional space."""


class DESpatialReferencesFolder(Parameter):
    """A folder on disk that stores coordinate systems."""


class DECoverage(Parameter):
    """A coverage dataset, which is a proprietary data model for storing geographic features as points, arcs, and polygons with associated feature attribute tables."""


class DECoverageFeatureClasses(Parameter):
    """A coverage feature class, such as point, arc, node, route, route system, section, polygon, and region."""


class DEType(Parameter):
    """A dataset visible in ArcCatalog."""


class GPDataFile(Parameter):
    """A data file."""


class DERemoteDatabaseFolder(Parameter):
    """The database connection folder in ArcCatalog."""


class DEDatasetType(Parameter):
    """A collection of related data, usually grouped or stored together."""


class GPDate(Parameter):
    """A date value."""


class DEDbaseTable(Parameter):
    """Attribute data stored in dBASE format."""


class GP3DADecimate(Parameter):
    """A subset of nodes of a TIN to create a generalized version of that TIN."""


class GPDiagramLayer(Parameter):
    """A diagram layer."""


class DEDiskConnection(Parameter):
    """An access path to a data storage device."""


class GPDouble(Parameter):
    """Any floating-point number stored as a double precision, 64-bit value."""


class GPEncryptedString(Parameter):
    """An encrypted string for passwords."""


class GPEnvelope(Parameter):
    """The coordinate pairs that define the minimum bounding rectangle in which the data source resides."""


class GPEvaluationScale(Parameter):
    """The scale value range and increment value applied to inputs in a weighted overlay operation."""


class GPExtent(Parameter):
    """The coordinate pairs that define the minimum bounding rectangle (x-minimum, y-minimum and x-maximum, y-maximum) of a data source. All coordinates for the data source are within this boundary."""


class GPSAExtractValues(Parameter):
    """An extract values parameter."""


class DEFeatureClass(Parameter):
    """A collection of spatial data with the same shape type: point, multipoint, polyline, and polygon."""


class DEFeatureDataset(Parameter):
    """A collection of feature classes that share a common geographic area and the same spatial reference system."""


class GPFeatureLayer(Parameter):
    """A reference to a feature class, including symbology and rendering properties."""


class GPFeatureRecordSetLayer(Parameter):
    """Interactive features that draw the features when the tool is run."""


class Field(Parameter):
    """A column in a table that stores the values for a single attribute."""


class GPFieldInfo(Parameter):
    """The details about a field in a field map."""


class GPFieldMapping(Parameter):
    """A collection of fields in one or more input tables."""


class DEFile(Parameter):
    """A file on disk."""


class DEFolder(Parameter):
    """A location on disk where data is stored."""


class GPRasterFormulated(Parameter):
    """A raster surface whose cell values are represented by a formula or constant."""


class GPSAFuzzyFunction(Parameter):
    """The algorithm used in fuzzification of an input raster."""


class DEGeodatasetType(Parameter):
    """A collection of data with a common theme in a geodatabase."""


class DEGeoDataServer(Parameter):
    """A coarse-grained object that references a geodatabase."""


class DEGeometricNetwork(Parameter):
    """A linear network represented by topologically connected edge and junction features. Feature connectivity is based on their geometric coincidence."""


class GPGALayer(Parameter):
    """A reference to a geostatistical data source, including symbology and rendering properties."""


class GPGASearchNeighborhood(Parameter):
    """The searching neighborhood parameters for a geostatistical layer are defined."""


class GPGAValueTable(Parameter):
    """A collection of data sources and fields that define a geostatistical layer."""


class DEGlobeServer(Parameter):
    """A Globe server."""


class DEGPServer(Parameter):
    """A geoprocessing server."""


class GPGraph(Parameter):
    """A graph."""


class GPGraphDataTable(Parameter):
    """A graph data table."""


class GPGroupLayer(Parameter):
    """A collection of layers that appear and act as a single layer. Group layers make it easier to organize a map, assign advanced drawing order options, and share layers for use in other maps."""


class GPSAHorizontalFactor(Parameter):
    """The relationship between the horizontal cost factor and the horizontal relative moving angle."""


class DEImageServer(Parameter):
    """An image service."""


class Index(Parameter):
    """A data structure used to speed the search for records in geographic datasets and databases."""


class GPINFOExpression(Parameter):
    """A syntax for defining and manipulating data in an INFO table."""


class GPArcInfoItem(Parameter):
    """An item in an INFO table."""


class DEArcInfoTable(Parameter):
    """A table in an INFO database."""


class GPInternetTiledLayer(Parameter):
    """An internet tiled layer."""


class GPKMLLayer(Parameter):
    """A KML layer."""


class GPKnowledgeGraphLayer(Parameter):
    """A layer that visualizes entities and relationships from a knowledge graph."""


class DELasDataset(Parameter):
    """A LAS dataset stores reference to one or more LAS files on disk as well as to additional surface features. A LAS file is a binary file that stores airborne lidar data."""


class GPLasDatasetLayer(Parameter):
    """A layer that references a LAS dataset on disk. This layer can apply filters on lidar files and surface constraints referenced by a LAS dataset."""


class GPLayer(Parameter):
    """A reference to a data source, such as a shapefile, coverage, geodatabase feature class, or raster, including symbology and rendering properties."""


class DELayer(Parameter):
    """A layer file stores a layer definition, including symbology and rendering properties."""


class GPLayout(Parameter):
    """A layout in an ArcGIS Pro project or a layout file (.pagx)."""


class GPLine(Parameter):
    """A shape, straight or curved, defined by a connected series of unique x,y-coordinate pairs."""


class GPLinearUnit(Parameter):
    """A linear unit type and value such as meter or feet."""


class GPLong(Parameter):
    """An integer number value."""


class GPMDomain(Parameter):
    """A range of lowest and highest possible value for m-coordinates."""


class GPMap(Parameter):
    """An ArcGIS Pro map."""


class DEMapServer(Parameter):
    """A map server."""


class GPMapServerLayer(Parameter):
    """A map server layer."""


class DEMosaicDataset(Parameter):
    """A collection of raster and image data that allows you to store, view, and query the data. It is a data model in the geodatabase used to manage a collection of raster datasets (images) stored as a catalog and viewed as a mosaicked image."""


class GPMosaicLayer(Parameter):
    """A layer that references a mosaic dataset."""


class GPSANeighborhood(Parameter):
    """The shape of the area around each cell used to calculate statistics."""


class NAClassFieldMap(Parameter):
    """The mapping between location properties in a Network Analyst layer (such as stops, facilities, and incidents) and a point feature class."""


class GPNAHierarchySettings(Parameter):
    """A hierarchy attribute that divides hierarchy values of a network dataset into three groups using two integers. The first integer sets the ending value of the first group; the second integer sets the beginning value of the third group."""


class GPNALayer(Parameter):
    """A group layer used to express and solve network routing problems. Each sublayer held in memory in a Network Analyst layer represents some aspect of the routing problem and the routing solution."""


class GPNetworkDataSource(Parameter):
    """A network data source can be a local dataset specified using its catalog path or a layer from a map, or it can be a URL to a portal."""


class DENetworkDataset(Parameter):
    """A collection of topologically connected network elements (edges, junctions, and turns), derived from network sources and associated with a collection of network attributes."""


class GPNetworkDatasetLayer(Parameter):
    """A reference to a network dataset, including symbology and rendering properties."""


class NetworkTravelMode(Parameter):
    """A dictionary of travel mode objects."""


class GPOrientedImageryLayer(Parameter):
    """A collection of camera locations with image metadata."""


class GPPairwiseWeightsTable(Parameter):
    """A table of a pairwise comparison matrix containing evaluations for each of the pairs for the input variables."""


class DECadastralFabric(Parameter):
    """A dataset used for the storage, maintenance, and editing of a continuous surface of connected parcels or parcel network."""


class GPCadastralFabricLayer(Parameter):
    """A layer referencing a parcel fabric on disk. This layer works as a group layer, organizing a set of related layers under a single layer."""


class GPPoint(Parameter):
    """A pair of x,y-coordinates."""


class GPPolygon(Parameter):
    """A connected sequence of x,y-coordinate pairs in which the first and last coordinate pair are the same."""


class DEPrjFile(Parameter):
    """A file storing coordinate system information for spatial data."""


class GPSAGDBEnvPyramid(Parameter):
    """Specifies whether pyramids are built."""


class GPSARadius(Parameter):
    """The surrounding points that are used for interpolation."""


class GPRandomNumberGenerator(Parameter):
    """The seed and the generator to use when creating random values."""


class DERasterBand(Parameter):
    """A layer in a raster dataset."""


class GPRasterCalculatorExpression(Parameter):
    """A raster calculator expression."""


class GPRasterDataLayer(Parameter):
    """A raster data layer."""


class DERasterDataset(Parameter):
    """A single dataset built from one or more rasters."""


class GPRasterLayer(Parameter):
    """A reference to a raster, including symbology and rendering properties."""


class GPSAGDBEnvStatistics(Parameter):
    """Specifies whether raster statistics will be built."""


class GPRasterBuilder(Parameter):
    """Raster data is added to a mosaic dataset by specifying a raster type. The raster type identifies metadata, such as georeferencing, acquisition date, and sensor type, with a raster format."""


class GPRecordSet(Parameter):
    """An interactive table. Provide the table values when the tool is run."""


class DERelationshipClass(Parameter):
    """The details about the relationship between objects in the geodatabase."""


class GPSARemap(Parameter):
    """A table that defines how raster cell values are reclassified."""


class GPRouteMeasureEventProperties(Parameter):
    """The fields on a table that describe events measured by a linear reference route system."""


class GPSceneServiceLayer(Parameter):
    """A scene service layer."""


class DESchematicDataset(Parameter):
    """A collection of schematic diagram templates and schematic feature classes that share the same application domain, for example, water or electrical."""


class DESchematicDiagram(Parameter):
    """A schematic diagram."""


class DESchematicDiagramClass(Parameter):
    """A schematic diagram class."""


class DESchematicFolder(Parameter):
    """A schematic folder."""


class GPSchematicLayer(Parameter):
    """A composite layer composed of feature layers based on the schematic feature classes associated with the template on which the schematic diagram is based."""


class GPSASemiVariogram(Parameter):
    """The distance and direction representing two locations used to quantify autocorrelation."""


class DEServerConnection(Parameter):
    """A server connection."""


class DEShapeFile(Parameter):
    """Spatial data in shapefile format."""


class GPSpatialReference(Parameter):
    """The coordinate system used to store a spatial dataset, including the spatial domain."""


class GPSQLExpression(Parameter):
    """A syntax for defining and manipulating data from a relational database."""


class GPString(Parameter):
    """A text value."""


class GPStringHidden(Parameter):
    """A string that is masked by asterisk characters. The text is not encrypted when used in scripting."""


class DETable(Parameter):
    """Tabular data."""


class GPTableView(Parameter):
    """A representation of tabular data for viewing and editing purposes stored in memory or on disk."""


class GPTerrainLayer(Parameter):
    """A reference to a terrain, including symbology and rendering properties. It's used to draw a terrain."""


class DETextfile(Parameter):
    """A text file."""


class GPSAGDBEnvTileSize(Parameter):
    """The width and height of data stored in block."""


class GPSATimeConfiguration(Parameter):
    """The time periods used for calculating solar radiation at specific locations."""


class GPTimeUnit(Parameter):
    """A time unit type and value such as minutes or hours."""


class DETin(Parameter):
    """A vector data structure that partitions geographic space into contiguous, nonoverlapping triangles. The vertices of each triangle are sample data points with x-, y-, and z-values."""


class GPTinLayer(Parameter):
    """A reference to a TIN, including topological relationships, symbology, and rendering properties."""


class DETool(Parameter):
    """A geoprocessing tool."""


class DEToolbox(Parameter):
    """A geoprocessing toolbox."""


class GPSATopoFeatures(Parameter):
    """Features that are input to the interpolation."""


class DETopology(Parameter):
    """A topology that defines and enforces data integrity rules for spatial data."""


class GPTopologyLayer(Parameter):
    """A reference to a topology, including symbology and rendering properties."""


class GPTrajectoryLayer(Parameter):
    """A layer that references a trajectory dataset."""


class GPSATransformationFunction(Parameter):
    """A Spatial Analyst transformation function."""


class DEUtilityNetwork(Parameter):
    """A utility network."""


class GPUtilityNetworkLayer(Parameter):
    """A utility network layer."""


class GPValueTable(Parameter):
    """A collection of columns of values."""


class GPVariant(Parameter):
    """A data value that can contain any basic type: Boolean, date, double, long, and string."""


class GPVectorLayer(Parameter):
    """A vector tile layer."""


class GPSAVerticalFactor(Parameter):
    """The relationship between the vertical cost factor and the vertical, relative moving angle."""


class DEVPFCoverage(Parameter):
    """Spatial data stored in Vector Product Format."""


class DEVPFTable(Parameter):
    """Attribute data stored in Vector Product Format."""


class DEWCSCoverage(Parameter):
    """Web Coverage Service (WCS) is an open specification for sharing raster datasets on the web."""


class GPSAWeightedOverlayTable(Parameter):
    """A table with data to combine multiple rasters by applying a common measurement scale of values to each raster, weighing each according to its importance."""


class GPSAWeightedSum(Parameter):
    """Data for overlaying several rasters, each multiplied by their given weight and summed."""


class DEWMSMap(Parameter):
    """A Web Map Service (WMS) specification map."""


class DEWorkspace(Parameter):
    """A container such as a geodatabase or folder."""


class GPXYDomain(Parameter):
    """A range of lowest and highest possible values for x,y-coordinates."""


class GPZDomain(Parameter):
    """A range of lowest and highest possible values for z-coordinates."""
