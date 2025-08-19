"""Cursor Helper module"""

from typing import (
    TypedDict,
    Required,
    Literal,
    NamedTuple,
    TYPE_CHECKING,
)

from arcpy import (
    SpatialReference,
    Geometry,
    Polygon,
    PointGeometry,
    Polyline,
    Multipoint,
    Extent,
)

if TYPE_CHECKING:
    from arcpy.da import (
        SpatialRelationship,
        SearchOrder,
    )
else:
    SpatialRelationship = None
    SearchOrder = None

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

GeometryType = Geometry | Polygon | PointGeometry | Polyline | Multipoint

class WhereClause:
    def __init__(self, where_clause: str) -> None:
        self.where_clause =  where_clause
    def __repr__(self) -> str:
        return self.where_clause

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
    prefix: str|None
    postfix: str|None

class SearchOptions(TypedDict, total=False):
    """Optional parameters for SearchCursors
    
    Arguments:
        where_clause (str): A SQL query that is inserted after the SQL `WHERE` (`SELECT {prefix} {fields} FROM {table} WHERE {where_clause} {postfix}...`)
        spatial_reference (str | int | SpatialReference): Perform an on the fly projection of the yielded geometry to this reference
        explode_to_points (bool): Return a row per vertex in each feature (e.g. `[SHAPE, 'eric', 'idle'] -> [Point, 'eric', 'idle'], [Point, 'eric', 'idle'], ...`)
        sql_clause (SQLClause): A tuple of SQL queries that is inserted after the SQL 
            `SELECT` and `WHERE` clauses (`SELECT {prefix} {fields} FROM {table} WHERE {where_clause} {postfix}...`)
        datum_transformation (str): The transformation to use during projection if there is a datum difference between the feature projection and the 
            target SpatialReference (you can use `arcpy.ListTransformations` to find valid transformations)
        spatial_filter (Geometry): A shape that will be used to test each feature against using the specified `spatial_relationship` (`'INTERSECTS'`)
            by default.
        spatial_relationship (SpatialRelationship): The type of relationship with the `spatial_filter` to test for in each row. Only rows with shapes
            that match this relationship will be yielded.
        search_order (SearchOrder): Run the `where_clause {sql_clause}` (`'ATTRIBUTEFIRST'` default) or `spatial_filter` (`'SPATIALFIRST'`) first.
            This can be used to optimize a cursor. If you have a complex `where_clause`, consider switching to `'SPATIALFIRST'` to cut down on the number
            of records that the `where_clause` runs for. These two operations are done as seperate SQL operations and `JOINED` in the result

    Returns:
        ( dict ): A dictionary with the populated keys

    Usage:
        ```python
            >>> options = SearchOptions(where_clause='OBJECTID > 10')
            >>> not_first_ten = feature_class.get_tuples(['NAME', 'LENGTH'], **options)
            >>> print(list(not_first_ten))
            [('cleese', 777), ('idle', 222), ('gilliam', 111), ...]
        
    """
    where_clause: str
    spatial_reference: str | int | SpatialReference
    explode_to_points: bool
    sql_clause: SQLClause
    datum_transformation: str
    spatial_filter: GeometryType | Extent
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
    #skip_nulls: bool
    #null_value: dict[str, Any]
    datum_transformation: str
    explicit: bool
    spatial_filter: GeometryType | Extent
    spatial_relationship: SpatialRelationship
    search_order: SearchOrder

FieldType = Literal[
    'SHORT',
    'LONG',
    'BIGINTEGER',
    'FLOAT',
    'DOUBLE',
    'TEXT',
    'DATE',
    'DATEHIGHPRECISION',
    'DATEONLY',
    'TIMEONLY',
    'TIMESTAMPOFFSET',
    'BLOB',
    'GUID',
    'RASTER',
]

class Field(TypedDict, total=False):
    """Field Representation
    
    Args:
        field_table (FieldType): The type of the field (required)
        field_precision (int): The precision (digits) of numeric fields (default: database determined)
        field_scale (int): The number of decimal places for floating point fields (default: database determined)
        field_length (int): The maximum character count for `TEXT` fields (default: 255)
        field_alias (str): Human readable alias for fields with confusing internal names (optional)
        field_is_nullable (bool): Allow null values (default: `True`)
        field_is_required (bool): Field requires a value to be set (default: False)
        field_domain (str): Existing Domain name to bind to field (optional)
    """
    field_type: Required[FieldType]
    field_precision: int
    field_scale: int
    field_length: int
    field_alias: str
    field_is_nullable: Literal['NULLABLE', 'NON_NULLABLE']
    field_is_required: Literal['REQUIRED']
    field_domain: bool