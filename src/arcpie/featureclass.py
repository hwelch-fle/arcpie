from __future__ import annotations

# Typing imports
import arcpy.typing.describe as dt
from string import ascii_letters, digits

from functools import reduce
import json
from pprint import pformat
from tempfile import TemporaryDirectory

from collections.abc import (
    Iterable,
    Iterator,
    Callable,
    Sequence,
    Mapping,
    Generator,
)

from typing import (
    Any,
    TypeVar,
    Generic,
    Literal,
    TYPE_CHECKING,
    overload,
    Self,
)

# Arcpy imports
from arcpy.da import (
    Editor,
    SearchCursor,
    InsertCursor,
    UpdateCursor,
    ListSubtypes,
    Describe as Describe_da,
)

if TYPE_CHECKING:
    # Shadow cursors during type check with proper typing
    #from _types import (
    #    SearchCursor,
    #    InsertCursor,
    #    UpdateCursor,
    #)
    from arcpy.da import (
        SpatialRelationship,
    )
else:
    SpatialRelationship = None

from ._types import (
    AddRuleOpts,
    AlterRuleOpts,
    Subtype,
    AttributeRule,
    convert_dtypes,
    to_rule_add,
    to_rule_alter,
    convert_rule,
)

from arcpy import (
    Polygon,
    Extent,
    Describe, #type:ignore
    SpatialReference,
    Exists,
    EnvManager,
)

from arcpy.management import (
    CopyFeatures,  #type:ignore
    DeleteField, #type:ignore
    AddField, #type:ignore
    RecalculateFeatureClassExtent, #type:ignore
    SelectLayerByAttribute, #type: ignore
    AddAttributeRule, #type: ignore
    AlterAttributeRule, #type: ignore
    DeleteAttributeRule, #type: ignore
    EnableAttributeRules, #type: ignore
    DisableAttributeRules, #type: ignore
    ReorderAttributeRule, #type: ignore
)

from arcpy._mp import ( 
    Layer,
    Map,
    Table as TableLayer, # Alias
)

from typing import (
    Unpack,
)

# Standardlib imports
from contextlib import (
    contextmanager,
)

from pathlib import (
    Path,
)

# Library imports
from .cursor import (
    SearchOptions, 
    InsertOptions, 
    UpdateOptions,
    SQLClause,
    WhereClause,
    Field,
    ShapeToken,
    FeatureToken,
    FeatureTokens,
    TableToken,
    TableTokens,
    GeometryType,
)

FieldName = str #| FeatureToken
"""Alias for string that specifies the function needs a valid fieldname"""

_T = TypeVar('_T')

def count(featureclass: FeatureClass[Any, Any] | Iterator[Any]) -> int:
    """Get the record count of a FeatureClass
    
    Args:
        featureclass (FeatureClass | Iterator): The FeatureClass or Iterator/view to count
    
    Example:
        ```python
        >>> fc = FeatureClass[PointGeometry]('MyFC')
        >>> count(fc)
        1000
        >>> count(fc[where('1=0')])
        0
        >>> boundary = next(FeatureClass[Polygon]('Boundaries').shapes)
        >>> count(fc[boundary])
        325
        ```
    """
    # The __len__() method of FeatureClass only iterates
    # object ID values so this is a small optimisation we can do
    if isinstance(featureclass, FeatureClass):
        return len(featureclass)
    
    return sum(1 for _ in featureclass)

def extract_singleton(vals: Sequence[Any] | Any) -> Any | Sequence[Any]:
    """Helper function to allow passing single values to arguments that expect a tuple
    
    Args:
        vals (Sequence[Any] | Any): The values to normalize based on item count
    
    Returns:
        ( Sequence[Any] | Any  ): The normalized sequence
    """
    # String sequences are returned directly
    if isinstance(vals, str):
        return vals
    
    # Singleton sequences are flattened to the first value
    if len(vals) == 1:
        return vals[0]
    
    # Default to returning the arg
    return vals

def as_dict(cursor: SearchCursor | UpdateCursor) -> Iterator[RowRecord]:
    """Take a Cusrsor object and yield rows from it 
    
    Args:
        cursor (SearchCursor | UpdateCursor): The cursor to convert to a RowRecord iterator
        
    Yields:
        Iterator[RowRecord]
        
    Example:
        ```python
        >>> for row in as_dict(SearchCursor('table', ['Name', 'City']))
        ...     print(f'{row["Name"]} lives in {row["City"]}')
        Dave lives in New York City
        Robert lives in Kansas City
        ...
    """
    yield from (dict(zip(cursor.fields, row)) for row in cursor)

def format_query_list(vals: Iterable[Any]) -> str:
    """Format a list of values into a SQL list"""
    if isinstance(vals, (str , int)):
        return f"{vals}"
    return ','.join([f"{val}" for val in vals])

def norm(val: Any) -> str:
    """Normalize a value for SQL query (wrap strings in single quotes)"""
    if isinstance(val, str):
        return f"'{val}'"
    return val

def where(where_clause: str) -> WhereClause:
    """Wrap a string in a WhereClause object to use with indexing
    
    Args:
        where_clause (str): A where clause string to mark as a clause
    
    Returns:
        WhereClause
        
    Example:
        ```python
        >>> for row in features[where('SHAPE_LENGTH > 10')]:
        ...     print(row)
        {'OBJECTID': 1, 'SHAPE_LENGTH': 11}
        {'OBJECTID': 2, 'SHAPE_LENGTH': 34}
        {'OBJECTID': 3, 'SHAPE_LENGTH': 78}
        ...
        ```
    """
    return WhereClause(where_clause)

def filter_fields(*fields: FieldName) -> Callable[[FilterFunc[RowRecord]], FilterFunc[RowRecord]]:
    """Decorator for filter functions that limits fields checked by the SearchCursor
    
    Args:
        *fields (FieldName): Varargs for the fields to limit the filter to
    
    Returns:
        (FilterFunc): A filter function with a `fields` attribute added
        Used with FeatureClass.filter to limit columns
    
    Note:
        Iterating filtered rows using a decorated filter will limit available columns inside the 
        context of the filter. This should only be used if you need to improve performance of a 
        filter and don't care about the fields not included in the `filter_fields` decorator:
    
        Example:
            ```python
            >>> @filter_fields('Name', 'Age')
            >>> def age_over_21(row):
            ...     return row['Age'] > 21
            ...
            >>> for row in feature_class[age_over_21]:
            ...     print(row)
            ...
            {'Name': 'John', 'Age': 23}
            {'Name': 'Terry', 'Age': 42}
            ...
            >>> for row in feature_class:
            ...     print(row)
            ...
            {'Name': 'John', 'LastName': 'Cleese', 'Age': 23}
            {'Name': 'Graham', 'LastName': 'Chapman', 'Age': 18}
            {'Name': 'Terry', 'LastName': 'Gilliam', 'Age': 42}
            ...
            ```

    Note:
        You can achieve field filtering using the `FeatureClass.fields_as` context manager as well. 
        This method adds a level of indentation and can be more extensible:
        
        Example:
            ```python
            >>> def age_over_21(row):
            ...     return row['Age'] > 21
            ...
            >>> with feature_class.fields_as('Name', 'Age'):
            ...     for row in feature_class[age_over_21]:
            ...         print(row)
            ...
            {'Name': 'John', 'Age': 23}
            {'Name': 'Terry', 'Age': 42}
            ```
        Since the inspected fields live in the same code block as the filter that uses them, you can 
        easily add the fields in one place. This method is preferred for data manipulation operations 
        while counting operations can use the decorated filter to cut down on boilerplate.
    """
    def _filter_wrapper(func: FilterFunc):
        setattr(func, 'fields', fields)
        return func
    return _filter_wrapper

def valid_field(fieldname: FieldName) -> bool:
    """Validate a fieldname"""
    return not (
            # Has characters
            len(fieldname) == 0
            # Is under 160 characters
            or len(fieldname) > 160
            # Doesn't start with a number
            or fieldname[0] in digits 
            # Only has alphanum and underscore
            or not set(fieldname).issubset(ascii_letters + digits + '_')
            # Doesn't have reserved prefix
            or any(fieldname.startswith(reserved) for reserved in ('gdb_', 'sde_', 'delta_'))
        )

RowRecord = dict[FieldName, Any]
"""Alias for a dictionary of fieldnames and field values"""
if TYPE_CHECKING: # Using some 3.13 features here

    _GeometryType = TypeVar('_GeometryType', bound=GeometryType, default=GeometryType)
    # Optional Schema to use for typing records
    _Schema = TypeVar('_Schema', bound=Mapping[str, Any], default=RowRecord)

    FilterFunc = Callable[[_Schema], bool]
    """The expected type signature for function indexing"""
else:
    _Schema = TypeVar('_Schema')
    _GeometryType = TypeVar('_GeometryType')
    FilterFunc = Callable[[RowRecord], bool]

class Table(Generic[_Schema]):
    """A Wrapper for ArcGIS Table objects"""
    
    Tokens = TableTokens
    
    def __init__(
            self, path: str|Path,
            *,
            search_options: SearchOptions|None=None, 
            update_options: UpdateOptions|None=None, 
            insert_options: InsertOptions|None=None,
            clause: SQLClause|None=None,
            where: str|None=None,
        ) -> None:
        self._path = str(path)
        self._clause = clause or SQLClause(None, None)
        self._search_options = search_options or SearchOptions()
        self._insert_options = insert_options or InsertOptions()
        self._update_options = update_options or UpdateOptions()
        
        # Override
        if where:
            self._search_options['where_clause'] = where
            self._update_options['where_clause'] = where
        
        self._layer: Layer|None=None
        self._in_edit_session=False
        self._fields: tuple[TableToken | str, ...]|None=None

    # rw Properties
    
    @property
    def search_options(self) -> SearchOptions:
        """Default SearchCursor options"""
        return self._search_options.copy()
    
    @search_options.setter
    def search_options(self, search_options: SearchOptions) -> None:
        """Default SearchCursor options setter"""
        self._search_options = search_options or SearchOptions()

    @property
    def insert_options(self) -> InsertOptions:
        """Default InsertCursor options"""
        return self._insert_options.copy()
    
    @insert_options.setter
    def insert_options(self, insert_options: InsertOptions) -> None:
        """Default InsertCursor options setter"""
        self._insert_options = insert_options or InsertOptions()

    @property
    def update_options(self) -> UpdateOptions:
        """Default UpdateCursor options"""
        return self._update_options.copy() # pyright: ignore[reportReturnType]
    
    @update_options.setter
    def update_options(self, update_options: UpdateOptions) -> None:
        """Default UpdateCursor options setter"""
        self._update_options = update_options or UpdateOptions()

    @property
    def clause(self) -> SQLClause:
        """Default SQLClause"""
        return self._clause

    @clause.setter
    def clause(self, clause: SQLClause) -> None:
        """Set a feature level SQL clause on all Insert and Search operations
        
        This clause is overridden by all Option level clauses
        """
        self._clause = clause

    @property
    def layer(self) -> Layer|None:
        """A Layer object for the FeatureClass/Table if one is bound"""
        return self._layer

    @layer.setter
    def layer(self, layer: Layer) -> None:
        """Set a layer object for the Table or FeatureClass, layer datasource must be this FeatureClass!"""
        if layer.dataSource != self.path:
            raise ValueError(f'Layer: {layer.name} does not source to {self.name} Table or FeatureClass at {self.path}!')
        self._layer = layer
    
    # ro Properties

    @property
    def path(self) -> str:
        """The filepath of the FeatureClass/Table"""
        return self._path

    @property
    def describe(self) -> dt.Table:
        """Access the arcpy.Describe object for the `Table` or `FeatureClass`"""
        return Describe(self.path) #type:ignore (Will be dt.Table or FeatureClass)

    @property
    def da_describe(self) -> dict[str, Any]:
        """Access the da.Describe dictionary for the `Table` or `FeatureClass`"""
        return Describe_da(self.path)
    
    @property
    def workspace(self) -> str:
        """Get the workspace of the `Table` or `FeatureClass`"""
        return self.describe.workspace.catalogPath

    @property
    def name(self) -> str:
        """The common name of the FeatureClass/Table"""
        return self.describe.name

    @property
    def oid_field_name(self) -> str:
        """ObjectID fieldname (ususally FID or OID or ObjectID)"""
        return self.describe.OIDFieldName

    @property
    def subtype_field(self) -> str | None:
        """The Subtype field (ususally SUBTYPE or SUBTYPE_CODE, etc.)"""
        if not self.subtypes:
            return None
        return list(self.subtypes.values()).pop()['SubtypeField']

    @property
    def fields(self) -> tuple[TableToken | str, ...]:
        """Tuple of all fieldnames in the Table or FeatureClass with `OID@` as first"""
        if not self._fields:
            exclude = (self.oid_field_name)
            replace = ('OID@',)
            _fields = ()
            with self.search_cursor('*') as c:
                _fields = c.fields
            self._fields = replace + tuple((f for f in _fields if f not in exclude))
        return self._fields

    @property
    def np_dtypes(self):
        """Numpy dtypes for each field"""
        return self.search_cursor(*self.fields)._dtype # pyright: ignore[reportPrivateUsage]

    @property
    def py_types(self) -> dict[str, type]:
        """Get a mapping of fieldnames to python types for the Table"""
        return convert_dtypes(self.np_dtypes)

    @property
    def subtypes(self) -> dict[int, Subtype]:
        """Result of ListSubtypes, mapping of code to Subtype object"""
        return ListSubtypes(self.path) # type:ignore

    @property
    def editor(self) -> Editor:
        """Get an Editor manager for the Table or FeatureClass
        Will set multiuser_mode to True if the feature can version
        """
        return Editor(self.workspace, multiuser_mode=self.describe.canVersion)

    @property
    def attribute_rules(self) -> AttributeRuleManager:
        """Get an `AttributeRuleManager` object bound to the Table/FeatureClass"""
        return AttributeRuleManager(self)

    # Option Resolvers (kwargs -> Options Object -> Table or FeatureClass Options)
    
    def _resolve_search_options(self, options: SearchOptions|None, overrides: SearchOptions) -> SearchOptions:
        """Combine all provided SearchOptions into one dictionary"""
        return {
            'sql_clause': self.clause or SQLClause(None, None), 
            **self.search_options, 
            **(options or {}), 
            **overrides
        }

    def _resolve_insert_options(self, options: InsertOptions|None, overrides: InsertOptions) -> InsertOptions:
        """Combine all provided InsertOptions into one dictionary"""
        return {**self.insert_options, **(options or {}), **overrides}

    def _resolve_update_options(self, options: UpdateOptions|None, overrides: UpdateOptions) -> UpdateOptions:
        """Combine all provided UpdateOptions into one dictionary"""
        return {
            'sql_clause': self.clause or SQLClause(None, None), 
            **self.update_options, 
            **(options or {}), 
            **overrides
        }

    # Cursor Handlers
    
    def search_cursor(self, *field_names: FieldName,
                      search_options: SearchOptions|None=None, 
                      **overrides: Unpack[SearchOptions]) -> SearchCursor:
        """Get a `SearchCursor` for the `Table` or `FeatureClass`
        Supplied search options are resolved by updating the base `Table` or `FeatureClass` Search options in this order:

        `**overrides['kwarg'] -> search_options['kwarg'] -> self.search_options['kwarg']`

        This is implemented using unpacking operations with the lowest importance option set being unpacked first

        `{**self.search_options, **(search_options or {}), **overrides}`
        
        With direct key word arguments (`**overrides`) shadowing all other supplied options. This allows a FeatureClass to
        be initialized using a base set of options, then a shared SearchOptions set to be applied in some contexts,
        then a direct keyword override to be supplied while never mutating the base options of the FeatureClass.
        
        Args:
            field_names (str | Iterable[str]): The column names to include from the `Table` or `FeatureClass`
            search_options (SearchOptions|None): A `SeachOptions` instance that will be used to shadow
                `search_options` set on the `Table` or `FeatureClass`
            **overrides ( Unpack[SeachOptions] ): Additional keyword arguments for the cursor that shadow 
                both the `seach_options` variable and the `Table` or `FeatureClass` instance `SearchOptions`
        
        Returns:
            ( SearchCursor ): A `SearchCursor` for the `Table` or `FeatureClass` instance that has all supplied options
                resolved and applied
                
        Example:
            ```python
                >>> cleese_search = SearchOptions(where_clause="NAME = 'John Cleese'")
                >>> idle_search = SearchOptions(where_clause="NAME = 'Eric Idle'")
                >>> monty = Table or FeatureClass('<path>', search_options=cleese_search)
                >>> print(list(monty.search_cursor('NAME')))
                [('John Cleese',)]
                >>> print(list(monty.search_cursor('NAME', search_options=idle_search)))
                [('Eric Idle', )]
                >>> print(list(monty.search_cursor('NAME', search_options=idle_search)), where_clause="NAME = Graham Chapman")
                [('Graham Chapman', )]
            ```
        In this example, you can see that the keyword override is the most important. The fact that the other searches are
        created outside initialization allows you to store common queries in one place and update them for all cursors using 
        them at the same time, while still allowing specific instances of a cursor to override those shared/stored defaults.
        """
        return SearchCursor(self.path, field_names, **self._resolve_search_options(search_options, overrides))

    def insert_cursor(self, *field_names: FieldName,
                      insert_options: InsertOptions|None=None, 
                      **overrides: Unpack[InsertOptions]) -> InsertCursor:
        """See `Table.search_cursor` doc for general info. Operation of this method is identical but returns an `InsertCursor`"""
        return InsertCursor(self.path, field_names, **self._resolve_insert_options(insert_options, overrides))

    def update_cursor(self, *field_names: FieldName,
                    update_options: UpdateOptions|None=None, 
                    **overrides: Unpack[UpdateOptions]) -> UpdateCursor:
        """See `Table.search_cursor` doc for general info. Operation of this method is identical but returns an `UpdateCursor`"""
        return UpdateCursor(self.path, field_names, **self._resolve_update_options(update_options, overrides))

    def row_updater(self, *field_names: FieldName,
                    strict: bool=False,
                    update_options: UpdateOptions|None=None, 
                    **overrides: Unpack[UpdateOptions]) -> Generator[_Schema, _Schema|None, None]:
        """A Bi-Directional generator that yields rows and updates them with the sent value
        
        Note:
            This method will assume the full provided schema if there is one, so make sure you keep track of
            any applied field filters.
        
        Args:
            fields (FieldName|str): The fields to include in the update operation (default: All)
            stict (bool): Raise a KeyError if an invalid fieldname is passed, otherwise drop invalid updates (default: False)
            update_options (UpdateOptions): Additional context to pass to the UpdateCursor as a dictionary
            **overrides (UpdateOptions): Additional context to pass to the UpdateCursor as keyword arguments
        
        Example:
            ```python
            >>> updater = fc.row_updater()
            >>> for row in updater:
            ...     if row['Name'] = 'No Name':
            ...         row['Name'] = None
            ...         updater.send(row)
            ```
        """
        with self.update_cursor(*(field_names or self.fields), update_options=update_options, **overrides) as cur:
            for row in self.as_dict(cur):
                upd = yield row
                
                if strict and (invalid := set(upd) - (set(row))):
                    raise KeyError(f'{invalid} fields not found in {self.name}')
                
                if upd is not None and isinstance(row, dict):
                    row = {upd.get(k, row[k]) for k in row}
                    cur.updateRow(list(row.values()))
    
    @contextmanager
    def updater(self, *fields: FieldName, strict: bool=False):
        """A wrapper around `row_updater` that allows use as a context manager
        
        This simplifies the interaction with the `row_updater` method by allowing inline declaration
        of the generator. For most simple update operations, this manager should work well. 
        
        Args:
            fields (FieldName|str): The fields to include in the update operation (default: All)
            stict (bool): Raise a KeyError if an invalid fieldname is passed, otherwise drop invalid updates (default: False)
        
        Example:
            >>> with fc.editor, fc.updater() as upd:
            ...     for row in upd:
            ...         row['Name'] = 'Dave'
            ...         upd.send(row)
        """
        try:
            yield self.row_updater(*(fields or self.fields), strict=strict)
        finally:
            pass
    
    # Localize as_dict for internal typing of _Schema var
    def as_dict(self, cursor: SearchCursor | UpdateCursor) -> Iterator[_Schema]:
        yield from as_dict(cursor) # pyright: ignore[reportReturnType]

    if TYPE_CHECKING:
        GroupIter = Iterator[tuple[Any, ...] | Any]
        GroupIdent = tuple[Any, ...] | Any
    def group_by(self, group_fields: Sequence[FieldName] | FieldName, return_fields: Sequence[FieldName] | FieldName ='*') -> Iterator[tuple[GroupIdent, GroupIter]]:
        """Group features by matching field values and yield full records in groups
        
        Args:
            group_fields (FieldOpt): The fields to group the data by
            return_fields (FieldOpt): The fields to include in the output record (`'*'` means all and is default)
        Yields:
            ( Iterator[tuple[tuple[FieldName, ...], Iterator[tuple[Any, ...] | Any]]] ): A nested iterator of groups and then rows
        
        Example:
            ```python
            >>> # With a field group, you will be able to unpack the tuple
            >>> for group, rows in fc.group_by(['GroupField1', 'GroupField2'], ['ValueField1', 'ValueField2', ...]):
            ...     print(group)
            ...     for v1, v2 in rows:
            ...        if v1 > 10:
            ...            print(v2)
            (GroupValue1A, GroupValue1B)
            valueA
            valueB
            ...
            >>> # With a single field, you will have direct access to the field values   
            >>> for group, district_populations in fc.group_by(['City', 'State'], 'Population'):
            >>>         print(f"{group}: {sum(district_populations)}")
            (New York, NY): 8260000
            (Boston, MA): 4941632
            ...
            ```
        """

        # Parameter Validations
        if isinstance(group_fields, str):
            group_fields = (group_fields,)
        if return_fields == '*':
            return_fields = self.fields
        if isinstance(return_fields, str):
            return_fields = (return_fields,)
        if len(group_fields) < 1 or len(return_fields) < 1:
            raise ValueError("Group Fields and Return Fields must be populated")
        
        group_fields = list(group_fields)
        return_fields = list(return_fields)
        _all_fields = group_fields + return_fields
        for group in self.distinct(group_fields):
            group_key = {field : value for field, value in zip(group_fields, group)}
            where_clause = " AND ".join(f"{field} = {norm(value)}" for field, value in group_key.items())
            if '@' not in where_clause: # Handle valid clause (no tokens)
                with self.search_cursor(*return_fields, where_clause=where_clause) as group_cur:
                    yield (extract_singleton(group), (extract_singleton(row) for row in group_cur))
            else: # Handle token being passed by iterating a cursor and checking values directly
                for row in filter(lambda row: all(row[k] == group_key[k] for k in group_key), self[set(_all_fields)]):
                    yield (extract_singleton(group), (row.pop(k) for k in return_fields)) # type: ignore (TypedDict Generic causes issues)

    def distinct(self, distinct_fields: Iterable[FieldName] | FieldName) -> Iterator[tuple[Any, ...]]:
        """Yield rows of distinct values
        
        Args:
            distinct_fields (FieldOpt): The field or fields to find distinct values for.
                Choosing multiple fields will find all distinct instances of those field combinations
        
        Yields:
            ( tuple[Any, ...] ): A tuple containing the distinct values (single fields will yield `(value, )` tuples)
        """
        clause = SQLClause(prefix=f'DISTINCT {format_query_list(distinct_fields)}', postfix=None)
        try:
            yield from (value for value in self.search_cursor(*distinct_fields, sql_clause=clause))
        except RuntimeError: # Fallback when DISTINCT is not available or fails with Token input
            yield from sorted(set(self.get_tuples(distinct_fields)))

    def get_records(self, field_names: Iterable[FieldName] | FieldName, **options: Unpack[SearchOptions]) -> Iterator[_Schema]:
        """Generate row dicts with in the form `{field: value, ...}` for each row in the cursor

        Args:
            field_names (str | Iterable[str]): The columns to iterate
            **options (Unpack[SearchOptions]): Additional options to pass on to the cursor
        Yields 
            ( dict[str, Any] ): A mapping of fieldnames to field values for each row
        """
        with self.search_cursor(*field_names, **options) as cur:
            yield from self.as_dict(cur)

    def get_tuples(self, field_names: Iterable[FieldName] | FieldName, **options: Unpack[SearchOptions]) -> Iterator[tuple[Any, ...]]:
        """Generate tuple rows in the for (val1, val2, ...) for each row in the cursor
        
        Args:
            field_names (str | Iterable[str]): The columns to iterate
            **options (SearchOptions): Additional parameters to pass to the SearchCursor
        """
        with self.search_cursor(*field_names, **options) as cur:
            yield from cur

    def insert_record(self, record: _Schema, ignore_errors: bool=False) -> int | None:
        """Insert a single record into the table"""
        if missing_fields := set(record.keys()).difference(self.fields):
            if ignore_errors:
                return None
            else:
                raise ValueError(f'{missing_fields} not in {self.fields}')
        with self.insert_cursor(*record.keys()) as cur:
            return cur.insertRow(list(record.values()))

    def insert_records(self, records: Iterable[_Schema] , ignore_errors: bool=False) -> Iterator[int]:
        """Provide am iterable of records to insert
        Args:
            records (Iterable[RowRecord]): The sequence of records to insert
            ignore_errors (bool): Ignore per-row errors and continue. Otherwise raise KeyError (default: True)
        
        Returns:
            ( Iterator[int] ): Returns the OIDs of the newly inserted rows

        Raises:
            ( KeyError ): If the records have varying keys or the keys are not in the Table or FeatureClass
            
        Example:
            ```python
            >>> new_rows = [
            ...    {'first': 'John', 'last': 'Cleese', 'year': 1939}, 
            ...    {'first': 'Michael', 'last': 'Palin', 'year': 1943}
            ... ]
            >>> print(fc.insert_rows(new_rows))
            (2,3)
            
            >>> # Insert all shapes from fc into fc2
            >>> fc2.insert_rows(fc.get_records(['first', 'last', 'year']))
            (1,2)
            ```
        """
        yield from filter(None, (self.insert_record(record, ignore_errors=ignore_errors) for record in records))
 
    def delete_identical(self, field_names: Iterable[FieldName] | FieldName) -> dict[int, int]:
        """Delete all records that have matching field values
        
        Args:
            field_names (Sequence[FieldName] | FieldName): The fields used to define an identical feature
        
        Returns:
            (dict[int, int]): A dictionary of count of identical features deleted per feature
            
        Note:
            Insertion order takes precidence unless the Table or FeatureClass is ordered. The first feature found
            by the cursor will be maintained and all subsequent matches will be removed
        """
        # All
        if isinstance(field_names, str):
            field_names = [field_names]
            
        unique: dict[int, tuple[Any]] = {}
        deleted: dict[int, int] = {}
        with self.update_cursor('OID@', *field_names) as cur:
            for row in cur:
                oid: int = row[0]
                row = tuple(row[1:])
                for match_id, match_row in unique.items():
                    if all(a == b for a, b in zip(row, match_row)):
                        match = match_id
                        break
                else:
                    match = False
                
                if not match:
                    unique[oid] = row
                
                else:
                    deleted.setdefault(match, 0)
                    deleted[match] += 1
                    cur.deleteRow()
        return deleted
                
    def filter(self, func: FilterFunc[_Schema], invert: bool=False) -> Iterator[_Schema]:
        """Apply a function filter to rows in the Table or FeatureClass

        Args:
            func (Callable[[dict[str, Any]], bool]): A callable that takes a 
                row dictionary and returns True or False
            invert (bool): Invert the function. Only yield rows that return `False`
        
        Yields:
            ( dict[str, Any] ): Rows in the Table or FeatureClass that match the filter (or inverted filter)

        Example:
            ```python
            >>> def area_filter(row: dict) -> bool:
            >>>     return row['Area'] >= 10

            >>> for row in fc:
            >>>     print(row['Area'])
            1
            2
            10
            <etc>
            
            >>> for row in fc.filter(area_filter):
            >>>     print(row['Area'])
            10
            11
            90
            <etc>
            ```

        """
        if hasattr(func, 'fields'): # Allow decorated filters for faster iteration (see `filter_fields`)
            with self.fields_as(*getattr(func, 'fields')):
                yield from (row for row in self if func(row) == (not invert))
        else:
            yield from (row for row in self if func(row) == (not invert))

    # Data Operations
    
    def copy(self, workspace: str, options: bool=True) -> Self:
        """Copy this `Table` or `FeatureClass` to a new workspace
        
        Args:
            workspace (str): The path to the workspace
            options (bool): Copy the cursor options to the new `Table` or `FeatureClass` (default: `True`)
            
        Returns:
            (Table or FeatureClass): A `Table` or `FeatureClass` instance of the copied features
        
        Example:
            ```python
            >>> new_fc = fc.copy('workspace2')
            >>> new_fc == fc
            False
            ```
        """
        #name = Path(self.path).relative_to(Path(self.workspace))
        if Exists(copy_fc := Path(workspace) / self.name):
            raise ValueError(f'{self.name} already exists in {workspace}!')
        CopyFeatures(self.path, str(copy_fc))
        fc = self.__class__(str(copy_fc))
        if options:
            fc.search_options = self.search_options
            fc.update_options = self.update_options
            fc.insert_options = self.insert_options
            fc.clause = self.clause
        return fc

    def exists(self) -> bool:
        """Check if the Table or FeatureClass actually exists (check for deletion or initialization with bad path)"""
        return Exists(str(self))

    def has_field(self, fieldname: str) -> bool:
        """Check if the field exists in the featureclass or is a valid Token (@[TOKEN])"""
        return fieldname in self.fields or fieldname in self.Tokens

    def add_field(self, fieldname: str, field: Field|None=None, **options: Unpack[Field]) -> None:
        """Add a new field to a Table or FeatureClass, if no type is provided, deafault of `VARCHAR(255)` is used
        
        Args:
            fieldname (str): The name of the new field (must not start with a number and be alphanum or underscored)
            field (Field): A Field object that contains the desired field properties
            **options (**Field): Allow passing keyword arguments for field directly (Overrides field arg)

        Example:
            ```python
            >>> new_field = Field(
            ...     field_alias='Abbreviated Month',
            ...     field_type='TEXT',
            ...     field_length='3',
            ...     field_domain='Months_ABBR',
            ... )
            
            >>> print(fc.fields)
            ['OID@', 'SHAPE@', 'name', 'year']
            
            >>> fc['month'] = new_field
            >>> fc2['month'] = new_field # Can re-use a field definition 
            >>> print(fc.fields)
            ['OID@', 'SHAPE@', 'name', 'year', 'month']
            ```
        """
        if self.has_field(fieldname):
            raise ValueError(f'{self.name} already has a field called {fieldname}!')
        
        # Use provided field or default to 'TEXT' and override with kwargs
        field = {**(field or Field(field_type='TEXT')), **options}
        
        # Handle malformed Field arg
        field['field_type'] = field.get('field_type', 'TEXT')
        
        _option_kwargs = set(Field.__optional_keys__) | set(Field.__required_keys__)
        _provided = set(field.keys())
        
        if not _provided <= _option_kwargs:
            raise ValueError(f"Unknown Field properties provided: {_provided - _option_kwargs}")
        
        if not valid_field(fieldname):
            raise ValueError(
                f"{fieldname} is invalid, fieldnames must not start with a number "
                "and must only contain alphanumeric characters and underscores"
            )
        
        with EnvManager(workspace=self.workspace):
            AddField(self.path, fieldname, **field)
            self._fields = None

    def add_fields(self, fields: dict[str, Field]) -> None:
        """Provide a mapping of fieldnames to Fields
        
        Args:
            fields (dict[str, Field]): A mapping of fieldnames to Field objects
            
        Example:
            ```python
            >>> fields = {'f1': Field(...), 'f2': Field(...)}
            >>> fc.add_fields(fields)
            >>> fc.fields
            ['OID@', 'SHAPE@', 'f1', 'f2']
            ```
        """
        for fieldname, field in fields.items():
            self.add_field(fieldname, field)

    def delete_field(self, fieldname: str) -> None:
        """Delete a field from a Table or FeatureClass
        
        Args:
            fieldname (str): The name of the field to delete/drop
        
        Example:
            ```python
            >>> print(fc.fields)
            ['OID@', 'SHAPE@', 'name', 'year', 'month']
            
            >>> del fc['month']
            >>> print(fc.fields)
            ['OID@', 'SHAPE@', 'name', 'year']
            >>> fc.delete_field('year')
            >>> print(fc.fields)
            ['OID@', 'SHAPE@', 'name']
            ```
        """
        if fieldname in self.Tokens:
            raise ValueError(f"{fieldname} is a Token and cannot be deleted!")
        if not self.has_field(fieldname):
            raise ValueError(f"{fieldname} does not exist in {self.name}")
        with EnvManager(workspace=self.workspace):
            DeleteField(self.path, fieldname)
            self._fields = None # Defer new field check to next access

    def delete_fields(self, fieldnames: Iterable[FieldName]) -> None:
        for fname in fieldnames:
            self.delete_field(fname)
    
    def clear(self) -> None:
        """Clear all records from the table"""
        with self.update_cursor(self.oid_field_name) as cur:
            for _ in cur:
                cur.deleteRow()
    
    def delete_where(self, clause: WhereClause|str) -> None:
        """Delete all records that match the provided where clause
        
        Args:
            clause (WhereClause|str): The SQL query that determines the records that will be deleted
        """
        with self.where(clause):
            self.clear()
    
    # Magic Methods
    
    def __bool__(self) -> Literal[True]:
        # Override __bool__ to prevent fallback to __len__
        return True
    
    _IndexableTypes = FieldName | set[FieldName] | list[FieldName] | tuple[FieldName, ...] | WhereClause | None
        
    @overload
    def __getitem__(self, field: tuple[FieldName, ...]) -> Iterator[tuple[Any, ...]]: ...
    @overload
    def __getitem__(self, field: list[FieldName]) -> Iterator[list[Any]]: ...
    @overload
    def __getitem__(self, field: set[FieldName]) -> Iterator[_Schema]: ...
    @overload
    def __getitem__(self, field: FieldName) -> Iterator[Any]: ...
    @overload
    def __getitem__(self, field: FilterFunc[_Schema]) -> Iterator[_Schema]: ...
    @overload
    def __getitem__(self, field: WhereClause) -> Iterator[_Schema]: ...
    @overload
    def __getitem__(self, field: None) -> Iterator[None]: ...
    def __getitem__(self, field: _IndexableTypes | FilterFunc[_Schema]) -> Iterator[Any]:
        """Handle all defined overloads using pattern matching syntax
        
        Args:
            field (str): Yield values in the specified column (values only)
            field (list[str]): Yield lists of values for requested columns (requested fields)
            field (tuple[str]): Yield tuples of values for requested columns (requested fields)
            field (set[str]): Yield dictionaries of values for requested columns (requested fields)
            field (FilterFunc): Yield rows that match function (all fields)
            field (WhereClause): Yield rows that match clause (all fields)

        Example:
            ```python
            >>> # Single Field
            >>> print(list(fc['field']))
            [val1, val2, val3, ...]
            
            >>> # Field Tuple
            >>> print(list(fc[('field1', 'field2')]))
            [(val1, val2), (val1, val2), ...]
             
            >>> # Field List
            >>> print(list(fc[['field1', 'field2']]))
            [[val1, val2], [val1, val2], ...]
            
            >>> # Field Set (Row mapping limited to only requested fields)
            >>> print(list(fc[{'field1', 'field2'}]))
            [{'field1': val1, 'field2': val2}, {'field1': val1, 'field2': val2}, ...]
            
            >>> # Last two options always return all fields in a mapping
            >>> # Filter Function (passed to Table.filter())
            >>> print(list(fc[lambda r: r['field1'] == target]))
            [{'field1': val1, 'field2': val2, ...}, {'field1': val1, 'field2': val2, ...}, ...]
             
            >>> # Where Clause (Use where() helper function or a WhereClause object)
            >>> print(list(fc[where('field1 = target')]))
            [{'field1': val1, 'field2': val2, ...}, {'field1': val1, 'field2': val2, ...}, ...]
            
            >>> # None (Empty Iterator)
            >>> print(list(fc[None]))

            ```
        """
        match field:
            # Field Requests
            case str():
                with self.search_cursor(field) as cur:
                    yield from (val for val, in cur)
            case tuple():
                with self.search_cursor(*field) as cur:
                    yield from (row for row in cur)
            case list():
                with self.search_cursor(*field) as cur:
                    yield from (list(row) for row in cur)
            case set():
                with self.search_cursor(*field) as cur:
                    yield from (row for row in self.as_dict(cur))
            case None:
                yield from () # This allows a side effect None to be used to get nothing

            # Conditional Requests
            case wc if isinstance(wc, WhereClause):
                if not wc.validate(self.fields):
                    raise KeyError(f'Invalid Where Clause: {wc}, fields not found in {self.name}')
                with self.search_cursor(*self.fields, where_clause=wc.where_clause) as cur:
                    yield from (row for row in self.as_dict(cur))
            case func if callable(func):
                yield from (row for row in self.filter(func))
            case _:
                raise KeyError(
                    f"Invalid option: `{field}` "
                    "Must be a WhereClause, filter functon, field, set of fields, list of fields, or tuple of fields"
                )

    @overload
    def get(self, field: tuple[FieldName, ...], default: _T) -> Iterator[tuple[Any, ...]] | _T: ...
    @overload
    def get(self, field: list[FieldName], default: _T) -> Iterator[list[Any]] | _T: ...
    @overload
    def get(self, field: set[FieldName], default: _T) -> Iterator[_Schema] | _T: ...
    @overload
    def get(self, field: FieldName, default: _T) -> Iterator[Any] | _T: ...
    @overload
    def get(self, field: FilterFunc[_Schema], default: _T) -> Iterator[_Schema] | _T: ...
    @overload
    def get(self, field: WhereClause, default: _T) -> Iterator[_Schema] | _T: ...
    @overload
    def get(self, field: None, default: _T) -> Iterator[None] | _T: ...
    def get(self, field: _IndexableTypes | FilterFunc[_Schema], default: _T=None) -> Iterator[Any] | _T:
        """Allow accessing the implemented indexes defined by `__getitem__` with a default shielding a raised `KeyError`
        
        Args:
            field (_Indexable_Types): The index to check (see `__getitem__` implementations)
            default (_T): A default to return when the indexing raises a `KeyError` or cursor field `RuntimeError` (default: None)
        
        Example:
            ```python
            >>> for name, age in fc[('Name', 'Age')]:
            >>>     print(name, age)
            ...
            KeyError "Name"
            ...
            
            >>> for name, age in fc.get(('Name', 'Age'), [])
            ```
        
        """
        try:
            return self[field]
        except (KeyError , RuntimeError) as e:
            if isinstance(e, RuntimeError) and 'Cannot find field' not in str(e):
                raise # Raise any non field related RuntimeErrors
            return default
    
    def __contains__(self, field: str) -> bool:
        """Implementation of contains that checks for a field existing in the `FeatureClass`
        """
        return field in self.fields

    def __iter__(self) -> Iterator[_Schema] | Iterator[Any]:
        """Iterate all rows in the Table or FeatureClass yielding mappings of field name to field value
        
        Note:
            It was decided to yield mappings because without specifying fields, it is up to the user
            to deal with the data as they see fit. Yielding tuples in an order that's not defined by
            the user would be confusing, so a mapping makes it clear exactly what they're accessing
            
        Note:
            When a single field is specified using the `fields_as` context, values will be yielded
        """ 
        with self.search_cursor(*self.fields) as cur:
            if len(self.fields) == 1:
                yield from (row for row, in cur)
            else:
                yield from self.as_dict(cur)

    def __len__(self) -> int:
        """Iterate all rows and count them. Only count with `self.search_options` queries.

        Note:
            The `__format__('len')` spec calls this function. So `len(fc)` and `f'{fc:len}'` are the same, 
            with the caveat that the format spec option returns a string

        Warning:
            This operation will traverse the whole dataset when called! You should not use it in loops:
            ```python
            # Bad
            for i, _ in enumerate(fc):
                print(f'{i}/{len(fc)}')

            # Good
            count = len(fc)
            for i, _ in enumerate(fc):
                print(f'{i}/{count}')
            ```
        """
        #return sum(1 for _ in self['OID@'])
        return sum(1 for _ in self.search_cursor('OID@'))

    def __repr__(self) -> str:
        """Provide a constructor string e.g. `Table or FeatureClass[Polygon]('path')`"""
        return f"{self.__class__.__name__}('{self.__fspath__()}')"

    def __str__(self) -> str:
        """Return the `Table` or `FeatureClass` path for use with other arcpy methods"""
        return self.__fspath__()

    def __eq__(self, other: Any) -> bool:
        """Determine if the datasource of two featureclass objects is the same"""
        return isinstance(other, self.__class__) and self.__fspath__() == other.__fspath__()

    def __format__(self, format_spec: str) -> str:
        """Implement format specs for string formatting a featureclass.

        Warning:
            The `{fc:len}` spec should only be used when needed. This spec will call `__len__` when 
            used and will traverse the entire Table or FeatureClass with applied SearchOptions each time it is 
            called. See: `__len__` doc for info on better ways to track counts in loops.

        Args:
            format_spec:  One of the options listed below (the `|` symbol is used to seperate aliases)

        Other Parameters:
            path|pth (str): Table or FeatureClass path
            len|length (str): Table or FeatureClass length (with applied SearchQuery)
            layer|lyr (str): Linked Table or FeatureClass layer if applicable (else `'None'`)
            shape|shp (str): Table or FeatureClass shape type
            units|unt (str): Table or FeatureClass linear unit name
            wkid|code (str): Table or FeatureClass WKID
            name|nm (str): Table or FeatureClass name
            fields|fld (str): Table or FeatureClass fields (comma seperated)

        Example:
            ```python
            >>> f'{fc:wkid}'
            '2236'
            >>> f'{fc:path}'
            'C:\\<FeaturePath>'
            >>> f'{fc:len}'
            '101'
            >>> f'{fc:shape}'
            'Polygon'
            ```
        """
        match format_spec:
            case 'path' | 'pth':
                return self.path
            case 'len' | 'length':
                return str(len(self))
            case 'layer' | 'lyr':
                return self.layer.longName if self.layer else 'None'
            case 'name' | 'nm':
                return self.name
            case 'fields' | 'flds':
                return ','.join(self.fields)
            case _:
                return str(self)

    def __fspath__(self) -> str:
        return str(Path(self.path).resolve())

    def __hash__(self) -> int:
        return hash(self.__fspath__())

    # Handle Fields
    
    def __delitem__(self, fieldname: str) -> None:
        self.delete_field(fieldname)

    def __setitem__(self, fieldname: str, field: Field) -> None:
        if fieldname in self.fields:
            raise ValueError(f'{fieldname} already exists in {self.name}')
        if not set(field.keys()).issubset([*Field.__optional_keys__, *Field.__required_keys__]):
            raise ValueError(f'Provided Field options are invalid, see `Field` from arcpie.cursor or arcpy for valid keys')
        self.add_field(fieldname, **field)

    # Context Managers
    
    @contextmanager
    def fields_as(self, *fields: FieldName):
        """Override the default fields for the Table or FeatureClass so all non-explicit Iterators will
        only yield these fields (e.g. `for row in fc: ...`)
        
        Args:
            *fields (FieldName): Varargs of the fieldnames to limit all unspecified Iterators to
        
        Example:
            ```python
            >>> with fc.fields_as('OID@', 'NAME'):
            ...     for row in fc:
            ...         print(row)
            {'OID@': 1, 'NAME': 'John'}
            {'OID@': 2, 'NAME': 'Michael'}
            ...
            >>> for row in fc:
            ...     print(row)
            {'OID@': 1, 'NAME': 'John', 'AGE': 75, 'ADDRESS': 123 Silly Walk}
            {'OID@': 2, 'NAME': 'Michael', 'AGE': 70, 'ADDRESS': 42 Dead Parrot Blvd}
            ...
            ```
        """
        # Allow passing a single field as a string `fc.fields_as('OID@')` to maintain
        # The call format of *Cursor objects
        _fields = self.fields
        self._fields = tuple(fields)
        try:
            yield self
        finally:
            self._fields = _fields
    
    @contextmanager
    def options(self,
                *, 
                strict: bool = False,
                search_options: SearchOptions|None=None, 
                update_options: UpdateOptions|None=None, 
                insert_options: InsertOptions|None=None, 
                clause: SQLClause|None=None):
        """Enter a context block where the supplied options replace the stored options for the `Table` or `FeatureClass`
        
        Args:
            strict (bool): If this is set to `True` the `Table` or `FeatureClass` will not fallback on existing options
                when set to `False`, provided options override existing options (default: `False`)
            search_options (SearchOptions): Contextual search overrides
            update_options (UpdateOptions): Contextual update overrides
            insert_options (InsertOptions): Contextual insert overrides
            clause (SQLClause): Contextual `sql_clause` override
        """
        _src_ops = self.search_options
        _upd_ops = self.update_options
        _ins_ops = self.insert_options
        _clause  = self.clause
        try:
            self._search_options = (
                self._resolve_search_options(_src_ops, search_options or {}) 
                if not strict
                else search_options or SearchOptions()
            )
            self._update_options = (
                self._resolve_update_options(_upd_ops, update_options or {})
                if not strict 
                else insert_options or UpdateOptions()
            )
            self._insert_options = (
                self._resolve_insert_options(_ins_ops, insert_options or {})
                if not strict 
                else insert_options or InsertOptions()
            )
            self._clause = (
                clause or _clause
                if not strict 
                else SQLClause(None, None)
            )
            yield self

        finally:
            self._search_options = _src_ops
            self._update_options = _upd_ops
            self.insert_options = _ins_ops
            self._clause = _clause

    @contextmanager
    def where(self, where_clause: WhereClause|str):
        """Apply a where clause to a Table or FeatureClass in a context

        Args:
            where_clause (WhereClause|str): The where clause to apply to the Table or FeatureClass
        
        Example:
            ```python
            >>> with fc.where("first = 'John'") as f:
            ...     for f in fc:
            ...         print(f)
            {'first': 'John', 'last': 'Cleese', 'year': 1939}

            >>> with fc.where('year > 1939'):
            ...     print(len(fc))
            5
            ... print(len(fc))
            6
            ```

        Note:
            This method of filtering a Table or FeatureClass will always be more performant than using the 
            `.filter` method. If you can achieve the filtering you want with a where clause, do it.
        """
        with self.options(
            search_options=SearchOptions(where_clause=str(where_clause)),
            update_options=UpdateOptions(where_clause=str(where_clause))):
            yield self

    # Mapping interfaces (These pass common `Layer` operations up to the Table or FeatureClass)
    def bind_to_layer(self, layer: Layer) -> None:
        """Update the provided layer's datasource to this Table or FeatureClass
        
        Args:
            layer (Layer): The layer to update connection properties for
        """
        layer.updateConnectionProperties(layer.dataSource, self.path) #type:ignore

    def add_to_map(self, map: Map, pos: Literal['AUTO_ARRANGE', 'BOTTOM', 'TOP']='AUTO_ARRANGE') -> None:
        """Add the featureclass to a map

        Note: 
            If the Table or FeatureClass has a layer, the bound layer will be added to the map. 
            Otherwise a default layer will be added. And the new layer will be bound to the Table or FeatureClass

        Args:
            map (Map): The map to add the featureclass to
        """
        if not self.layer:
            # Create a default layer, bind it, remove, and add back
            # with addLayer to match behavior with existing bound layer
            self.layer = map.addDataFromPath(self.path) #type:ignore (Always Layer)
            map.removeLayer(self.layer) #type:ignore (Incorrect Signature)
        map.addLayer(self.layer, pos) #type:ignore

    def select(self, method: Literal['NEW','DIFFERENCE','INTERSECT','SYMDIFFERENCE','UNION']='NEW') -> None:
        """If the Table or FeatureClass is bound to a layer, update the layer selection with the active SearchOptions
        
        Args:
            method: The method to use to apply the selection\n
                `DIFFERENCE`: Selects the features that are not in the current selection but are in the Table or FeatureClass.\n
                `INTERSECT`: Selects the features that are in the current selection and the Table or FeatureClass.\n
                `NEW`: Creates a new feature selection from the Table or FeatureClass.\n
                `SYMDIFFERENCE`: Selects the features that are in the current selection or the Table or FeatureClass but not both.\n
                `UNION`: Selects all the features in both the current selection and those in Table or FeatureClass.\n
        
        Note:
            Selection changes require the project file to be saved to take effect. 
        """
        if self.layer:
            _selected = list(self['OID@'])
            self.layer.setSelectionSet(_selected, method=method)
            try: # Try to select the layer in the active map
                if len(_selected) == 1:
                    _query = f'{self.oid_field_name} = {_selected.pop()})'
                if len(_selected) > 1:
                    _query = f'{self.oid_field_name} IN ({format_query_list(_selected)})'
                else:
                    return
                SelectLayerByAttribute(self.layer.longName, 'NEW_SELECTION', _query)
            except Exception:
                return
   
    def unselect(self) -> None:
        """If the Table or FeatureClass is bound to a layer, Remove layer selection
        
        Note:
            Selection changes require the project file to be saved to take effect.
        """
        if self.layer:
            self.layer.setSelectionSet(method='NEW')
            try: # Try to unselect the layer in the active map
                SelectLayerByAttribute(self.layer.longName, 'CLEAR_SELECTION')
            except Exception:
                return

    # Factory Constructors
    @classmethod
    def from_table(cls, table: TableLayer,
                   *,
                   ignore_selection: bool = False,
                   ignore_def_query: bool = False,) -> Table:
        """See `from_layer` for documentation, this is an alternative constructor that builds from a mp.Table object"""
        return Table.from_layer(table, ignore_selection=ignore_selection, ignore_def_query=ignore_def_query) # type: ignore (this won't break the interface)
    
    @classmethod
    def from_layer(cls, layer: Layer,
                   *,
                   ignore_selection: bool = False,
                   ignore_def_query: bool = False,) -> Table[Any]:
        """Build a Table or FeatureClass object from a layer applying the layer's current selection to the stored cursors
        
        Args:
            layer (Layer): The layer to convert to a Table or FeatureClass
            ignore_selection (bool): Ignore the layer selection (default: False)
            ignore_def_query (bool): Ignore the layer definition query (default: False)
        Returns:
            ( Table or FeatureClass ): The Table or FeatureClass object with the layer query applied
        """
        fc = cls(Path(layer.dataSource).resolve())
        
        selected_ids: set[int] | None = (
            layer.getSelectionSet() or None
            if not ignore_selection 
            else None
        )
        definition_query: str|None = (
            layer.definitionQuery or None
            if not ignore_def_query 
            else None
        )
        selection: str|None = (
            f"{fc.oid_field_name} IN ({format_query_list(selected_ids)})" 
            if selected_ids 
            else None
        )
        
        if (query_components := list(filter(None, [definition_query, selection]))):
            where_clause = ' AND '.join(query_components)
            fc.search_options = SearchOptions(where_clause=where_clause)
            fc.update_options = UpdateOptions(where_clause=where_clause)
            
        fc.layer = layer
        return fc
    
class FeatureClass(Table[_Schema], Generic[_GeometryType, _Schema]):
    """A Wrapper for ArcGIS FeatureClass objects
    
    Example:
        ```python
        >>> # Initialize FeatureClass with Geometry Type
        >>> point_features = FeatureClass[PointGeometry]('<feature_class_path>')
        >>> # Create a buffer Iterator
        >>> buffers = (pt.buffer(10) for pt in point_features.shapes)
        ... 
        >>> sr = SpatialReference(4206)
        >>> # Set a new spatial reference
        >>> with point_features.reference_as(sr):
        ...     # Consume the Iterator, but with the new reference
        ...     for buffer in buffers:
        ...        area = buffer.area
        ...        units = sr.linearUnitName
        ...        print(f"{area} Sq{units}")
        ```
    """

    Tokens = FeatureTokens

    def __init__(
            self, path: str|Path,
            *,
            search_options: SearchOptions|None=None, 
            update_options: UpdateOptions|None=None, 
            insert_options: InsertOptions|None=None,
            clause: SQLClause|None=None,
            where: str|None=None,
            shape_token: ShapeToken='SHAPE@'
        ) -> None:
        super().__init__(
            path=path, 
            search_options=search_options, update_options=update_options, insert_options=insert_options, 
            clause=clause, where=where
        )
        self._shape_token: ShapeToken = shape_token

    # rw Properties
    
    @property
    def shape_token(self) -> ShapeToken:
        """Set the default `SHAPE@??` token for iteration. Use `SHAPE@` for full shape (default: `SHAPE@`)"""
        return self._shape_token

    @shape_token.setter
    def shape_token(self, shape_token: ShapeToken) -> None:
        self._shape_token = shape_token

    # ro Properties

    @property
    def describe(self) -> dt.FeatureClass: # pyright: ignore[reportIncompatibleMethodOverride]
        """A describe object fort the FeatureClass"""
        return Describe(self.path) # type: ignore

    @property
    def shape_field_name(self) -> str:
        """The name for the base shape field of the FeatureClass"""
        return self.describe.shapeFieldName

    @property
    def fields(self) -> tuple[FieldName | FeatureToken, ...]:
        """Tuple of all fieldnames in the FeatureClass with `OID@` and `SHAPE@` as first 2"""
        if self._fields:
            return self._fields
        exclude = (self.oid_field_name, self.shape_field_name)
        replace = ('OID@', self.shape_token)
        _fields = ()
        with self.search_cursor('*') as c:
            _fields = c.fields
        self._fields = replace + tuple((f for f in _fields if f not in exclude))
        return self._fields

    @property
    def shapes(self) -> Iterator[_GeometryType]:
        """An iterator of feature shapes"""
        yield from ( shape for shape, in self.search_cursor('SHAPE@'))

    @property
    def spatial_reference(self):
        """The SpatialReference object for the FeatureClass"""
        return self.describe.spatialReference

    @property
    def units(self) -> str:
        """The unit name of the FeatureClass"""
        return self.spatial_reference.linearUnitName

    @property
    def extent(self) -> Extent:
        """Get the stored extent of the FeatureClass"""
        return self.describe.extent

    @property
    def py_types(self) -> dict[str, type]:
        """Get a mapping of the field types for the FeatureClass"""
        _types = convert_dtypes(self.np_dtypes)
        if 'SHAPE@' in _types and len(self) > 0:
                _types['SHAPE@'] = type(next(self.shapes))
        return _types
    # Data Operations
    
    @overload
    def footprint(self, buffer: float) -> Polygon | None: ...
    @overload
    def footprint(self, buffer: None) -> _GeometryType | None: ...
    def footprint(self, buffer: float|None=None) -> _GeometryType | Polygon | None:
        """Merge all geometry in the featureclass using current SelectionOptions into a single geometry object to use 
        as a spatial filter on other FeatureClasses
        
        Args:
            buffer (float | None): Optional buffer (in feature units, respects projection context) to buffer by (default: None)

        Returns:
            (GeometryType | None): A merged Multi-Geometry of all feature geometries or `None` if no features in FeatureClass
        """
        if len(self) == 0:
            return None

        def merge(acc: _GeometryType | Polygon, nxt: _GeometryType | Polygon) -> _GeometryType | Polygon:
            # Return type of union is Geometry for all types which is incorrect, it is Polygon
            if buffer:
                return acc.union(nxt.buffer(buffer)) # pyright: ignore[reportReturnType]
            else:
                return acc.union(nxt) # pyright: ignore[reportReturnType]
        
        # Consume the shape generator popping off the first shape and applying the buffer, 
        # Then buffering each additional shape and merging it into the accumulator (starting with _first)
        _shapes = self.shapes
        for _first in _shapes:
            break
        else:
            return None
        
        if buffer:
            _first = _first.buffer(buffer)
        
        return reduce(merge, _shapes, _first)
    
    def recalculate_extent(self) -> None:
        """Recalculate the FeatureClass Extent"""
        RecalculateFeatureClassExtent(self.path, 'STORE_EXTENT')

    # Magic Methods
    
    @overload
    def __getitem__(self, field: tuple[FieldName, ...]) -> Iterator[tuple[Any, ...]]: ...
    @overload
    def __getitem__(self, field: list[FieldName]) -> Iterator[list[Any]]: ...
    @overload
    def __getitem__(self, field: set[FieldName]) -> Iterator[_Schema]: ...
    @overload # Overload 'SHAPE@' for special case before FieldName (which it is a subset of)
    def __getitem__(self, field: Literal['SHAPE@']) -> Iterator[_GeometryType]: ...
    @overload
    def __getitem__(self, field: FieldName) -> Iterator[Any]: ...
    @overload
    def __getitem__(self, field: FilterFunc[_Schema]) -> Iterator[_Schema]: ...
    @overload
    def __getitem__(self, field: WhereClause) -> Iterator[_Schema]: ...
    @overload
    def __getitem__(self, field: None) -> Iterator[None]: ...
    @overload
    def __getitem__(self, field: GeometryType | Extent) -> Iterator[_Schema]: ...
    def __getitem__(self, field: Table._IndexableTypes | FilterFunc[_Schema] | Extent | GeometryType | Literal['SHAPE@']) -> Iterator[Any]:
        """Handle all defined overloads using pattern matching syntax
        
        Args:
            field (str): Yield values in the specified column (values only)
            field (list[str]): Yield lists of values for requested columns (requested fields)
            field (tuple[str]): Yield tuples of values for requested columns (requested fields)
            field (set[str]): Yield dictionaries of values for requested columns (requested fields)
            field (Geometry | Extent): Yield dictionaries of values for all features intersecting the specified shape
            field (FilterFunc): Yield rows that match function (all fields)
            field (WhereClause): Yield rows that match clause (all fields)

        Example:
            ```python
            >>> # Single Field
            >>> print(list(fc['field']))
            [val1, val2, val3, ...]
            
            >>> # Field Tuple
            >>> print(list(fc[('field1', 'field2')]))
            [(val1, val2), (val1, val2), ...]
             
            >>> # Field List
            >>> print(list(fc[['field1', 'field2']]))
            [[val1, val2], [val1, val2], ...]
            
            >>> # Field Set (Row mapping limited to only requested fields)
            >>> print(list(fc[{'field1', 'field2'}]))
            [{'field1': val1, 'field2': val2}, {'field1': val1, 'field2': val2}, ...]
            
            >>> # Last two options always return all fields in a mapping
            >>> # Filter Function (passed to FeatureClass.filter())
            >>> print(list(fc[lambda r: r['field1'] == target]))
            [{'field1': val1, 'field2': val2, ...}, {'field1': val1, 'field2': val2, ...}, ...]
             
            >>> # Where Clause (Use where() helper function or a WhereClause object)
            >>> print(list(fc[where('field1 = target')]))
            [{'field1': val1, 'field2': val2, ...}, {'field1': val1, 'field2': val2, ...}, ...]
             
            >>> # Shape Filter (provide a shape to use as a spatial filter on the rows)
            >>> print(list(fc[shape]))
            [{'field1': val1, 'field2': val2, ...}, {'field1': val1, 'field2': val2, ...}, ...]
            
            >>> # None (Empty Iterator)
            >>> print(list(fc[None]))
            ```
        """
        match field:
            case 'SHAPE@':
                yield from self.shapes
            case shape if isinstance(shape, Extent | GeometryType):
                with self.search_cursor(*self.fields, spatial_filter=shape) as cur:
                    yield from (row for row in self.as_dict(cur))
            case field if isinstance(field, str|set|list|tuple|Callable|WhereClause|None):
                yield from super().__getitem__(field)
            case _:
                raise KeyError(f'{type(field)}: {field}')
    
    @overload
    def get(self, field: tuple[FieldName, ...], default: _T) -> Iterator[tuple[Any, ...]] | _T: ...
    @overload
    def get(self, field: list[FieldName], default: _T) -> Iterator[list[Any]] | _T: ...
    @overload
    def get(self, field: set[FieldName], default: _T) -> Iterator[_Schema] | _T: ...
    @overload # Overload 'SHAPE@' for special case before FieldName (which it is a subset of)
    def get(self, field: Literal['SHAPE@'], default: _T) -> Iterator[_GeometryType] | _T: ...
    @overload
    def get(self, field: FieldName, default: _T) -> Iterator[Any] | _T: ...
    @overload
    def get(self, field: FilterFunc[_Schema], default: _T) -> Iterator[_Schema] | _T: ...
    @overload
    def get(self, field: WhereClause, default: _T) -> Iterator[_Schema] | _T: ...
    @overload
    def get(self, field: None, default: _T) -> Iterator[None] | _T: ...
    @overload
    def get(self, field: GeometryType | Extent, default: _T) -> Iterator[_Schema] | _T: ...
    def get(self, field: Table._IndexableTypes | FilterFunc[_Schema] | Extent | GeometryType | Literal['SHAPE@'], default: _T=None) -> Iterator[Any] | _T:
        """Allows safe indexing of a FeatureClass, see `Table.get` for more information"""
        try:
            return self[field]
        except (KeyError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and 'Cannot find field' in str(e):
                raise
            return default
    
    def __format__(self, format_spec: str) -> str:
        match format_spec:
            case 'shape' | 'shp':
                return self.describe.shapeType
            case 'wkid' | 'code':
                return str(self.spatial_reference.factoryCode)
            case 'unit':
                return self.spatial_reference.linearUnitName
            case _:
                return super().__format__(format_spec)

    # Context Managers
    
    @contextmanager
    def reference_as(self, spatial_reference: SpatialReference):
        """Allows you to temporarily set a spatial reference on SearchCursor and UpdateCursor objects within a context block
        
        Args:
            spatial_reference (SpatialReference): The spatial reference to apply to the cursor objects
        
        Yields:
            (self): Mutated self with search and update options set to use the provided spatial reference

        Example:
            ```python
            >>> sr = arcpy.SpatialReference(26971)
            >>> fc = FeatureClass[Polygon]('<fc_path>')
               
            >>> orig_shapes = list(fc.shapes)
               
            >>> with fc.project_as(sr):
            ...     proj_shapes = list(fc.shapes)
               
            >>> print(orig_shapes[0].spatialReference)
            SpatialReference(4326)
            
            >>> print(proj_shapes[0].spatialReference)
            SpatialReference(26971)
            ```
        """
        with self.options(
            search_options=SearchOptions(spatial_reference=spatial_reference), 
            update_options=UpdateOptions(spatial_reference=spatial_reference)):
            yield self

    @contextmanager
    def spatial_filter(self, spatial_filter: GeometryType | Extent, spatial_relationship: SpatialRelationship='INTERSECTS'):
        """Apply a spatial filter to the FeatureClass in a context
        
        Args:
            spatial_filter (Geometry | Extent): The geometry to use as a spatial filter
            spatial_relationship (SpatialRelationship): The relationship to check for (default: `INTERSECTS`)
        
        Example:
            ```python
            >>> with fc.spatial_filter(boundary) as f:
            ...     print(len(fc))
            100
            >>> print(len(fc))
            50000
            ```
        
        Note:
            Same as with `where`, this method will be much faster than any manual `filter` you can apply using python. 
            If you need to filter a FeatureClass by a spatial relationship, use this method, then do your expensive 
            `filter` operation on the reduced dataset

            ```python
            >>> def expensive_filter(rec):
            >>>     ...
            >>> with fc.spatial_filter(boundary) as local:
            >>>     for row in fc.filter(expensive_filter):
            >>>         ...
            ```
        """
        with self.options(
            search_options=SearchOptions(
                spatial_filter=spatial_filter, 
                spatial_relationship=spatial_relationship)):
            yield self

    # Factory Constructors

    @classmethod
    def from_layer(cls, layer: Layer,
                   *,
                   ignore_selection: bool = False,
                   ignore_def_query: bool = False,) -> FeatureClass[Any, Any]:
        """Build a FeatureClass object from a layer applying the layer's current selection to the stored cursors
        
        Args:
            layer (Layer): The layer to convert to a FeatureClass
            ignore_selection (bool): Ignore the layer selection (default: False)
            ignore_def_query (bool): Ignore the layer definition query (default: False)
        Returns:
            ( FeatureClass ): The FeatureClass object with the layer query applied
        """
        fc = cls(layer.dataSource)
        
        selected_ids: set[int] | None = (
            layer.getSelectionSet() or None
            if not ignore_selection 
            else None
        )
        definition_query: str|None = (
            layer.definitionQuery or None
            if not ignore_def_query 
            else None
        )
        selection: str|None = (
            f"{fc.oid_field_name} IN ({format_query_list(selected_ids)})" 
            if selected_ids 
            else None
        )
        
        if (query_components := list(filter(None, [definition_query, selection]))):
            where_clause = ' AND '.join(query_components)
            fc.search_options = SearchOptions(where_clause=where_clause)
            fc.update_options = UpdateOptions(where_clause=where_clause)
            
        fc.layer = layer
        return fc

class AttributeRuleManager:
    """Handler for interacting with AttributeRules on a FeatureClass or Table"""
    def __init__(self, parent: Table[Any]|FeatureClass[Any, Any]) -> None:
        self._parent = parent
            
    @property
    def names(self) -> list[str]:
        return list(self.rules.keys())
    
    @property
    def parent(self) -> Table[Any] | FeatureClass[Any, Any]:
        return self._parent 
    
    @property
    def rules(self) -> dict[str, AttributeRule]:
        return {
            rule['name']: AttributeRule(rule) 
            for rule in self._parent.da_describe['attributeRules']
        }
    
    def export_rules(self, out_dir: Path|str) -> None:
        """Write attribute rules out to a structured directory
        
        Args:
            out_dir (Path|str): The target directory to dump all attribute rules and configs to
        
        Note:
            out_dir -> fc_name -> [rule_name.cfg, rule_name.js]
        """
        out_dir = Path(out_dir)
        for rule_name, rule in self.rules.items():
            rule_name = rule_name.replace('/', '-') # Arc allows / in rulenames
            _script: str = str(rule.pop('scriptExpression', '')) # TypedDict has bugged pop typing
            out_file = out_dir / self._parent.name / rule_name
            out_file.parent.mkdir(exist_ok=True, parents=True)
            out_file.with_suffix('.js').write_text(_script)
            out_file.with_suffix('.cfg').write_text(json.dumps(rule, indent=2))
        return
    
    def import_rules(self, src_dir: Path|str, *, strict: bool=False, disable: bool=False) -> None:
        """Import attribute rules that were previously exported to the filesystem for editing
        
        Args:
            src_dir (Path|str): The directory that contains the `.cfg` and `.js` files for each rule
            strict (bool): Delete any attribute rules in the FeatureClass that do not have a matching file (default: False)
            disable (bool): Disable any attribute rules in the FeatureClass that do not have a matching file (default: False)
        
        Note:
            the `disable` option will be ignored if strict is not set
        """
        # Ensure that only the directory for the parent FC is accessed
        src_dir = Path(src_dir)
        if src_dir.stem != self.parent.name:
            src_dir = src_dir / self.parent.name
            
        _old_rules = {k: v.copy() for k,v in self.rules.items()}
        _imported_rule_names: set[str] = set()
        rule_config: AttributeRule = {'name': 'UNINITIALIZED'} # type: ignore
        try:
            for cfg in src_dir.glob('*.cfg'):
                # Grab base config and attach script sidecar
                rule_config: AttributeRule = json.loads(cfg.read_text(encoding='utf-8'))
                rule_script = cfg.with_suffix('.js').read_text(encoding='utf-8')
                rule = rule_config.copy()
                rule['scriptExpression'] = rule_script

                # Let the __setitem__ logic handle the rule (alter/add)
                self[rule['name']] = rule
                _imported_rule_names.add(rule['name'])

            if strict and (to_remove := set(self.names).difference(_imported_rule_names)):
                if disable:
                    self.disable_attribute_rules(list(to_remove))
                else:
                    self.delete_attribute_rules(list(to_remove))
        except Exception as e:
            # Revert the import if an Exception is rasied
            for rule_name, rule in _old_rules.items():
                if rule_name in _imported_rule_names:
                    self[rule_name] = rule
            
            # Remove rules
            if (to_remove := set(_old_rules).difference(self.names)):
                self.delete_attribute_rules(list(to_remove))
            
            e.add_note(f"{rule_config['name']} failed to import")
            e.add_note(f'Config: {pformat(convert_rule(rule_config))}')
            e.add_note(f'Transaction reverted for {_imported_rule_names} in {self.parent.name}')
            raise e # Raise the Exception
    
    def sync(self, target: FeatureClass[Any]|Table) -> None:
        """Sync the rules in this FeatureClass/Table instance with those of another overwriting 
        the current ruleset with the targeted ruleset
        
        Args:
            target (FeatureClass|Table): The target ruleset to overwrite the current rules with
        """
        # Use existing import functionality
        with TemporaryDirectory() as temp:
            target.attribute_rules.export_rules(temp)
            self.import_rules(temp)
    
    def add_attribute_rule(self, **rule: Unpack[AddRuleOpts]) -> None:
        
        # The AddAttributeRule function requires subtype codes to be converted to names
        # Since AlterAttributeRule does not accept subtypes
        _subtypes: list[str] = []
        for subtype in rule.get('subtype', []):
            if int(subtype) in self.parent.subtypes:
                _subtypes.append(self.parent.subtypes[int(subtype)]['Name'])
        if _subtypes:
            rule['subtype'] = _subtypes
        
        AddAttributeRule(self._parent.path, **rule)
    
    def alter_attribute_rule(self, evaluation_order: int | None=None, **rule: Unpack[AlterRuleOpts]) -> None:
        if evaluation_order: # Handle reorder
            ReorderAttributeRule(self._parent.path, rule['name'], evaluation_order)
        if rule:
            AlterAttributeRule(self._parent.path, **rule)
    
    def delete_attribute_rule(self, rule_name: str) -> None:
        DeleteAttributeRule(self._parent.path, rule_name)
        
    def delete_attribute_rules(self, rule_names: Sequence[str]) -> None:
        DeleteAttributeRule(self._parent.path, rule_names)
        
    def disable_attribute_rule(self, rule_name: str) -> None:
        DisableAttributeRules(self._parent.path, rule_name)
        
    def disable_attribute_rules(self, rule_names: Sequence[str]) -> None:
        DisableAttributeRules(self._parent.path, rule_names)
        
    def enable_attribute_rule(self, rule_name: str) -> None:
        EnableAttributeRules(self, rule_name)
        
    def enable_attribute_rules(self, rule_names: Sequence[str]) -> None:
        EnableAttributeRules(self, rule_names)
    
    def __iter__(self) -> Iterator[AttributeRule]:
        return iter(self.rules.values())
    
    def __getitem__(self, rule_name: str) -> AttributeRule:
        return self.rules[rule_name]
    
    def __contains__(self, name: str) -> bool:
        return name in self.names
    
    def __setitem__(self, rule_name: str, new_rule: AttributeRule) -> None:
        """The primary method for interacting with attribute rules
        
        The setitem override will take any dictionary that contains the keys expected by 
        the `AttributeRule` definition. Alteration or Addition is determined and applied 
        depending on the name of the rule and its state compared to the matching rule in 
        the current ruleset.
        
        Example:
            ```python
            >>> fc.attribute_rules.names
            ['Rule A', 'Rule B']
            >>> fc.attribute_rules['Rule A'] = {'isEnabled': False}
            ```
        """
        new_rule['name'] = rule_name
        current_rule = self.get(rule_name)
        is_enabled = new_rule.get('isEnabled', True)
        
        # Skip fields that are modified by the system
        skip_compare = {
            'id',
            'type',
            'requiredGeodatabaseClientVersion',
            'creationTime',
        }
        
        # Add a new rule
        if not current_rule:
            self.add_attribute_rule(**to_rule_add(new_rule))
            if not is_enabled:
                self.disable_attribute_rule(rule_name)
            return
        
        # Enable/Disable
        if is_enabled and not current_rule['isEnabled']:
            self.enable_attribute_rule(rule_name)
        elif not is_enabled and current_rule['isEnabled']:
            self.disable_attribute_rule(rule_name)
        is_enabled = current_rule['isEnabled']
        
        # Get Changes
        changes: dict[str, Any] = {
            setting: new_rule[setting]
            for setting in current_rule 
            if setting not in skip_compare
            and setting in new_rule
            and new_rule[setting] != current_rule[setting]
        }
        
        if not changes:
            return
        
        # Subtype change requires a re-build
        if 'subtypeCodes'in changes:
            self.delete_attribute_rule(rule_name)
            current_rule.update(new_rule)
            self.add_attribute_rule(**to_rule_add(current_rule))
        else:
            self.alter_attribute_rule(
                evaluation_order=changes.get('evaluatonOrder'),
                **to_rule_alter(new_rule)
            )
        
    def get(self, rule_name: str, default: _T=None) -> AttributeRule | _T:
        return self.rules.get(rule_name, default)

if __name__ == '__main__':
    pass