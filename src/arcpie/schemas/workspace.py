from __future__ import annotations
from typing import Any, TypedDict, Literal

# TODO: Add all string flags found in the wild here
ESRIClientVersion = Literal['10.0', '12.3']

ESRIWorkspaceType = Literal['esriLocalDatabaseWorkspace']
ESRIWorkspaceProgID = Literal["esriDataSourcesGDB.FileGDBWorkspaceFactory"]

ESRIDatasetType = Literal['DEWorkspace', 'esriDTFeatureClass', 'esriDTTable', 'esriDTTopology']
ESRISplitModel = Literal['esriSMUpdateInsert']

ESRIGeometryType = Literal['esriGeometryPolyline']
ESRIFieldType = Literal[
    'esriFieldTypeOID', 
    'esriFieldTypeString', 
    'esriFieldTypeInteger', 
    'esriFieldTypeDouble', 
    'esriFieldTypeGeometry',
]
ESRIFeatureType = Literal['esriFTSimple']
ESRIExtensionType = Literal['PropertySet']

ESRIDomainType = Literal['codedValue']
ESRIDomainMergePolicy = Literal['esriMPTDefaultValue']
ESRIDomainSplitPolicy = Literal['esriSPTDuplicate']

ESRIAttributeRuleType = Literal[
    'esriARTCalculation', 
    'esriARTValidation', 
    'esriARTConstraint',
]

ESRIAttributeRuleTrigger = Literal[
    'esriARTEUpdate', 
    'esriARTEInsert', 
    'esriARTEDelete',
]

# TODO: Add more
ESRITopologyRuleType = Literal[
    'esriTRTPointCoveredByLineEndpoint', 
    'esriTRTPointCoveredByLine',
    'esriTRTPointCoincidePoint',
    'esriTRTLineCoveredByLineClass', 
    'esriTRTLineEndpointCoveredByPoint',
    'esriTRTLineNoMultipart',
    'esriTRTLineNoSelfIntersect',
    'esriTRTLineNoSelfOverlap',
]

SchemaDomainCode = TypedDict(
    'SchemaDomainCode',
    {
        'name': str,
        'code': str
    }
)

SchemaDomain = TypedDict(
    'SchemaDomain',
    {
      'type': ESRIDomainType,
      'name': str,
      'description': str,
      'codedValues': list[SchemaDomainCode],
      'fieldType': ESRIFieldType,
      'mergePolicy': ESRIDomainMergePolicy,
      'splitPolicy': ESRIDomainSplitPolicy,
    }
)

SchemaSpatialRef = TypedDict(
    'SchemaSpatialRef',
    {
        'wkid': int,
        'latestWkid': int,
    }
)

SchemaGeometryDef = TypedDict(
    'SchemaGeometryDef',
    {
        'avgNumPoints': int,
        'geometryType': ESRIGeometryType,
        'hasM': bool,
        'hasZ': bool,
        'spatialReference': SchemaSpatialRef,
        'gridSize0': int,
    }
)

SchemaField = TypedDict(
    'SchemaField',
    {
        'name': str,
        'type': ESRIFieldType,
        'isNullable': bool,
        'length': int,
        'precision': int,
        'scale': int,
        'required': bool,
        'editable': bool,
        'aliasName': str,
        'geometryDef': SchemaGeometryDef,
        'domain': SchemaDomain
    },
    total = False
)
SchemaFieldArray = dict[Literal['fieldArray'], list[SchemaField]]


SchemaIndex = TypedDict(
    'SchemaIndex',
    {
        'name': str,
        'isUnique': bool,
        'isAscending': bool,
        'fields': SchemaFieldArray
    }
)
SchemaIndexArray = dict[Literal['indexArray'], list[SchemaIndex]]

SchemaPropertySet = TypedDict(
    'SchemaPropertySet',
    {
        'type': ESRIExtensionType,
        'propertySetItems': list[Any],
    }
)

SchemaFieldInfo = TypedDict(
    'SchemaFieldInfo',
    {
        'fieldName': str,
        'domainName': str,
        'defaultValue': Any,
    },
)

SchemaSubtype = TypedDict(
    'SchemaSubtype',
        {
            'subtypeName': str,
            'subtypeCode': int,
            'fieldInfos': list[SchemaFieldInfo]
        },
)
    
SchemaAttributeRule = TypedDict(
    'SchemaAttributeRule',
    {
        'id': int,
        'name': str,
        'type': ESRIAttributeRuleType,
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
        'triggeringEvents': list[ESRIAttributeRuleTrigger],
        'checkParameters': SchemaPropertySet,
        'category': int,
        'severity': int,
        'tags': str,
        'batch': bool,
        'requiredGeodatabaseClientVersion': ESRIClientVersion,
        'creationTime': int,
        'triggeringFields': list[str],
        'subtypeCodes': list[int]
    }
)

SchemaExtent = TypedDict(
    'SchemaExtent',
    {
        'xmin': float | Literal['NaN'],
        'ymin': float | Literal['NaN'],
        'xmax': float | Literal['NaN'],
        'ymax': float | Literal['NaN'],
        'spatialReference': SchemaSpatialRef
    }
)

SchemaRelationshipClasses = TypedDict(
    'SchemaRelationshipClasses',
    {
        'names': list[str],
    }
)

SchemaTopologyController = TypedDict(
    'SchemaTopologyController',
    {
        'topologyName': str,
        'weight': int,
        'xyRank': int,
        'zRank': int,
        'eventNotificationOnValidate': bool,
    }
)

SchemaParcelFabricController = TypedDict(
    'SchemaParcelFabricController',
    {
        # TODO: Find Schema
    }
)

SchemaTopologyRule = TypedDict(
    'SchemaTopologyRule',
    {
        'helpString': str,
        'ruleId': int,
        'name': str,
        'guid': str,
        'originClassID': int,
        'destinationClassID': int,
        'originSubtype': int,
        'triggerErrorEvents': bool,
        'allOriginSubtypes': bool,
        'allDestinationSubtypes': bool,
        'topologyRuleType': ESRITopologyRuleType
    }
)

SchemaDataset = TypedDict(
    'SchemaDataset',
    {
        'catalogPath': str,
        'name': str,
        'childrenExpanded': bool,
        'datasetType': ESRIDatasetType,
        'dsId': int,
        'versioned': bool,
        'canVersion': bool,
        'configurationKeyword': str,
        'requiredGeodatabaseClientVersion': ESRIClientVersion,
        'changeTracked': bool,
        'replicaTracked': bool,
        'hasOID': bool,
        'hasOID64': bool,
        'oidFieldName': str,
        'fields': SchemaFieldArray,
        'indexes': SchemaIndexArray,
        'clsId': str,
        'extClsId': str,
        'relationshipClassNames': SchemaRelationshipClasses,
        'aliasName': str,
        'modelName': str,
        'hasGlobalID': bool,
        'globalIdFieldName': str,
        'rasterFieldName': str,
        'extensionProperties': SchemaPropertySet,
        'subtypeFieldName': str,
        'defaultSubtypeCode': int,
        'subtypes': list[SchemaSubtype],
        'controllerMemberships': list[SchemaTopologyController | SchemaParcelFabricController], # TODO: Find all controllers
        'editorTrackingEnabled': bool,
        'creatorFieldName': str,
        'createdAtFieldName': str,
        'lastEditorFieldName': str,
        'editedAtFieldName': str,
        'isTimeInUTC': bool,
        'catalogID': str,
        'fieldFilteringEnabled': bool,
        'attributeRules': list[SchemaAttributeRule],
        'featureType': ESRIFeatureType,
        'shapeType': ESRIGeometryType,
        'shapeFieldName': str,
        'hasM': bool,
        'hasZ': bool,
        'hasSpatialIndex': bool,
        'areaFieldName': str,
        'lengthFieldName': str,
        'extent': SchemaExtent,
        'spatialReference': SchemaSpatialRef,
        'splitModel': ESRISplitModel,
        
        # Topology Keys (Move to a new TypedDict?)
        'layers': list[dict[Literal['layerId'], str]],
        'clusterTolerance': float,
        'zClusterTolerance': float,
        'maxGeneratedErrorCount': int,
        'topologyRules': list[SchemaTopologyRule]
        
    }
)

SchemaWorkspace = TypedDict(
    'SchemaWorkspace',
    {
        "datasetType": Literal["DEWorkspace"],
        "catalogPath": str,
        "name": str,
        "childrenExpanded": bool,
        "workspaceType": ESRIWorkspaceType,
        "workspaceFactoryProgID": ESRIWorkspaceProgID,
        "connectionString": str,
        "majorVersion": int,
        "minorVersion": int,
        "bugfixVersion": int,
        "realm": str,
        "maxAttributeRuleID": int,
        "domains": list[SchemaDomain],
        "datasets": list[SchemaDataset],
    }
)