"""Private module with type overrides for arcpy objects that make usage easier"""

from collections.abc import (
    Iterator, 
    Iterable, 
    Sequence,
)

from typing import (
    Self, 
    Any, 
    TypeVarTuple,
    TYPE_CHECKING,
    Unpack
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
_RowTs = TypeVarTuple('_RowTs', default=Unpack[tuple[Any, ...]])

class SearchCursor(Iterator[tuple[*_RowTs] | tuple[Any, ...]]):
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
    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType): ...
    def __next__(self) -> tuple[*_RowTs]: ...
    def __iter__(self) -> Iterator[tuple[*_RowTs]]: ...
    def next(self) -> tuple[*_RowTs]: ...
    def reset(self) -> None: ...
    def _as_narray(self) -> record: ...
    @property
    def fields(self) -> tuple[str, ...]: ...
    @property
    def _dtype(self) -> dtype: ...

class InsertCursor(Iterator[tuple[*_RowTs]]):
    _enable_simplify: bool = False
    def __init__(
        self,
        in_table: str | Layer,
        field_names: str | Iterable[str],
        datum_transformation: str | None = None,
        explicit: bool | None = False,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType): ...
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
    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType): ...
    def __next__(self) -> tuple[*_RowTs]: ...
    def __iter__(self) -> Iterator[tuple[*_RowTs]]: ...
    
from arcpy import Polygon
for oid, shape in SearchCursor[int, Polygon]('path', ["OID@", "SHAPE@"]):
    print(oid)
    x = shape.densify(0)
