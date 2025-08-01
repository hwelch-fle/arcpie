"""Cursor Helper module"""

from typing import (
    TypedDict,
    Any,
    Literal,
    NamedTuple,
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

class SQLClause(NamedTuple):
    """Wrapper for Cursor sql_clause attribute,
    
    Arguments:
        prefix (str): The SQL prefix to be prepended to the `FROM` part of the statment
        postfix (str): The SQL postfix that will be appended to the `WHERE` clause

    Format:
        `SELECT {prefix} {fields} FROM {table} WHERE {where_clause} {postfix}`

    Usage:
        ```python
            >>> five_longest = SQLClause(prefix='TOP 5', postfix='ORDER BY LENGTH DESC')
            >>> fc_result = feature_class.get_tuples(('NAME', 'LENGTH'), sql_clause=five_longest))
            >>> print(list(fc_result))
            [('foo', 1001), ('bar', 999), ('baz', 567), ('buzz', 345), ('bang', 233)]
        ```
    """
    prefix: str
    postfix: str

class SearchOptions(TypedDict, total=False):
    """Optional parameters for SearchCursors"""
    where_clause: str
    spatial_reference: str | int | SpatialReference
    explode_to_points: bool
    sql_clause: SQLClause
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
    sql_clause: SQLClause
    skip_nulls: bool
    null_value: dict[str, Any]
    datum_transformation: str
    explicit: bool
    spatial_filter: Geometry
    spatial_relationship: SpatialRelationship
    search_order: SearchOrder
