"""Cursor Helper module"""

from typing import (
    TypedDict,
    Any,
    Literal,
)

from arcpy import (
    SpatialReference,
    Geometry,
)

from arcpy.da import (
    SpatialRelationship,
    SearchOrder,
)

GeneralToken = Literal[
    'ANNO@',
    'GLOBALID@',
    'OID@',
    'SUBTYPE@',
]

ShapeToken = Literal[
    'SHAPE@XY',
    'SHAPE@XYZ',
    'SHAPE@TRUECENTROID',
    'SHAPE@X',
    'SHAPE@Y',
    'SHAPE@Z',
    'SHAPE@M',
    'SHAPE@JSON',
    'SHAPE@WKB',
    'SHAPE@WKT',
    'SHAPE@AREA',
    'SHAPE@LENGTH',
    'SHAPE@',
]

EditToken = Literal[
    'CREATED@',
    'CREATOR@',
    'EDITED@',
    'EDITOR@',
]

CursorToken = Literal[GeneralToken | ShapeToken | EditToken]
CursorTokens: tuple[CursorToken, ...] = CursorToken.__args__

class SearchOptions(TypedDict, total=False):
    """Optional parameters for SearchCursors"""
    where_clause: str
    spatial_reference: str | int | SpatialReference
    explode_to_points: bool
    sql_clause: tuple[str, str]
    datum_transformation: str
    spatial_filter: Geometry
    spatial_relationship: SpatialRelationship
    search_order: SearchOrder

class InsertOptions(TypedDict, total=False):
    """Optional parameters for InsertCursors"""
    datum_transformation: str
    explicit: bool

class UpdateOptions(TypedDict, total=False):
    """Optional parameters for UpdateCursors"""
    where_clause: str
    spatial_reference: str | int | SpatialReference
    explode_to_points: bool
    sql_clause: tuple[str, str]
    skip_nulls: bool
    null_value: dict[str, Any]
    datum_transformation: str
    explicit: bool
    spatial_filter: Geometry
    spatial_relationship: SpatialRelationship
    search_order: SearchOrder
