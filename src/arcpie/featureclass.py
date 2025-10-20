"""Sumbodule that helps wrap featureclass objects with useful methods"""

from __future__ import annotations

# Typing imports
import arcpy.typing.describe as dt
from string import ascii_letters, digits

from functools import reduce
import json

from collections.abc import (
    Iterable,
    Iterator,
    Callable,
    Sequence,
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
    Subtype,
    AttributeRule,
    convert_dtypes,
)

from arcpy import (
    Geometry,
    Polygon,
    Polyline,
    PointGeometry,
    Multipoint,
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
    ExportAttributeRules, #type: ignore
    ImportAttributeRules, #type: ignore
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

def count(featureclass: FeatureClass[Any] | Iterator[Any]) -> int:
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
    yield from ( dict(zip(cursor.fields, row)) for row in cursor )

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
    return WhereClause(where_clause)

def filter_fields(*fields: FieldName):
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
FilterFunc = Callable[[RowRecord], bool]
_GeometryType = TypeVar('_GeometryType', Geometry, Polygon, PointGeometry, Polyline, Multipoint, GeometryType)

class Table:
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
        return self._search_options.copy()
    
    @search_options.setter
    def search_options(self, search_options: SearchOptions) -> None:
        self._search_options = search_options or SearchOptions()

    @property
    def insert_options(self) -> InsertOptions:
        return self._insert_options.copy()
    
    @insert_options.setter
    def insert_options(self, insert_options: InsertOptions) -> None:
        self._insert_options = insert_options or InsertOptions()

    @property
    def update_options(self) -> UpdateOptions:
        return self._update_options.copy() # pyright: ignore[reportReturnType]
    
    @update_options.setter
    def update_options(self, update_options: UpdateOptions) -> None:
        self._update_options = update_options or UpdateOptions()

    @property
    def clause(self) -> SQLClause:
        return self._clause

    @clause.setter
    def clause(self, clause: SQLClause) -> None:
        """Set a feature level SQL clause on all Insert and Search operations
        
        This clause is overridden by all Option level clauses
        """
        self._clause = clause

    @property
    def layer(self) -> Layer|None:
        return self._layer

    @layer.setter
    def layer(self, layer: Layer) -> None:
        """Set a layer object for the Table or FeatureClass, layer datasource must be this feature class!"""
        if layer.dataSource != self.path:
            raise ValueError(f'Layer: {layer.name} does not source to {self.name} Table or FeatureClass at {self.path}!')
        self._layer = layer
    
    # ro Properties

    @property
    def path(self) -> str:
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
        return self.describe.name

    @property
    def oid_field_name(self) -> str:
        return self.describe.OIDFieldName

    @property
    def subtype_field(self) -> str | None:
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
        return self.search_cursor(self.fields)._dtype # pyright: ignore[reportPrivateUsage]

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
    def attribute_rules(self) -> dict[str, AttributeRule]:
        """Get FeatureClass/Table attribute rules access by name"""
        return {rule['name']: AttributeRule(rule) for rule in self.da_describe['attributeRules']}

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
    
    def search_cursor(self, field_names: FieldName | Sequence[FieldName],
                      *,
                      search_options: SearchOptions|None=None, 
                      **overrides: Unpack[SearchOptions]) -> SearchCursor:
        """Get a `SearchCursor` for the `Table` or `FeatureClass`
        Supplied search options are resolved by updating the base `Table` or `FeatureClass` Search options in this order:

        `**overrides['kwarg'] -> search_options['kwarg'] -> self.search_options['kwarg']`

        This is implemented using unpacking operations with the lowest importance option set being unpacked first

        `{**self.search_options, **(search_options or {}), **overrides}`
        
        With direct key word arguments (`**overrides`) shadowing all other supplied options. This allows a Feature Class to
        be initialized using a base set of options, then a shared SearchOptions set to be applied in some contexts,
        then a direct keyword override to be supplied while never mutating the base options of the feature class.
        
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

    def insert_cursor(self, field_names: FieldName | Sequence[FieldName],
                      *,
                      insert_options: InsertOptions|None=None, 
                      **overrides: Unpack[InsertOptions]) -> InsertCursor:
        """See `Table.search_cursor` doc for general info. Operation of this method is identical but returns an `InsertCursor`"""
        return InsertCursor(self.path, field_names, **self._resolve_insert_options(insert_options, overrides))

    def update_cursor(self, field_names: FieldName | Sequence[FieldName],
                      *,
                      update_options: UpdateOptions|None=None, 
                      **overrides: Unpack[UpdateOptions]) -> UpdateCursor:
        """See `Table.search_cursor` doc for general info. Operation of this method is identical but returns an `UpdateCursor`"""
        return UpdateCursor(self.path, field_names, **self._resolve_update_options(update_options, overrides))

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
                with self.search_cursor(return_fields, where_clause=where_clause) as group_cur:
                    yield (extract_singleton(group), (extract_singleton(row) for row in group_cur))
            else: # Handle token being passed by iterating a cursor and checking values directly
                for row in filter(lambda row: all(row[k] == group_key[k] for k in group_key), self[set(_all_fields)]):
                    yield (extract_singleton(group), (row.pop(k) for k in return_fields))

    def distinct(self, distinct_fields: Sequence[FieldName] | FieldName) -> Iterator[tuple[Any, ...]]:
        """Yield rows of distinct values
        
        Args:
            distinct_fields (FieldOpt): The field or fields to find distinct values for.
                Choosing multiple fields will find all distinct instances of those field combinations
        
        Yields:
            ( tuple[Any, ...] ): A tuple containing the distinct values (single fields will yield `(value, )` tuples)
        """
        clause = SQLClause(prefix=f'DISTINCT {format_query_list(distinct_fields)}', postfix=None)
        try:
            yield from (value for value in self.search_cursor(distinct_fields, sql_clause=clause))
        except RuntimeError: # Fallback when DISTINCT is not available or fails with Token input
            yield from sorted(set(self.get_tuples(distinct_fields)))

    def get_records(self, field_names: Sequence[FieldName] | FieldName, **options: Unpack[SearchOptions]) -> Iterator[RowRecord]:
        """Generate row dicts with in the form `{field: value, ...}` for each row in the cursor

        Args:
            field_names (str | Iterable[str]): The columns to iterate
            search_options (SearchOptions): A Search Options object
            **options (Unpack[SearchOptions]): Additional over
            search_options (SearchOptions): A Search Options object
            **options (Unpack[SearchOptions]): Additional over
        Yields 
            ( dict[str, Any] ): A mapping of fieldnames to field values for each row
        """
        with self.search_cursor(field_names, **options) as cur:
            yield from as_dict(cur)

    def get_tuples(self, field_names: Sequence[FieldName] | FieldName, **options: Unpack[SearchOptions]) -> Iterator[tuple[Any, ...]]:
        """Generate tuple rows in the for (val1, val2, ...) for each row in the cursor
        
        Args:
            field_names (str | Iterable[str]): The columns to iterate
            **options (SearchOptions): Additional parameters to pass to the SearchCursor
        """
        with self.search_cursor(field_names, **options) as cur:
            yield from cur

    def insert_records(self, records: Iterator[RowRecord] | Sequence[RowRecord], ignore_errors: bool=False) -> tuple[int, ...]:
        """Provide a list of records to insert
        Args:
            records (Iterable[RowRecord]): The sequence of records to insert
            ignore_errors (bool): Ignore per-row errors and continue. Otherwise raise KeyError (default: True)
        
        Returns:
            ( tuple[int] ): Returns the OIDs of the newly inserted rows

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
        # Always cast records to a list to prevent cursor race conditions, 
        # e.g. feature_class.insert_records(feature_class[where('SUBTYPE == 1')]) 
        # would insert infinite records since the search cursor trails the insert cursor.
        records = list(records)
        if not records:
            return tuple()
        
        rec_fields: list[str] = list(records[0].keys())
        def rec_filter(rec: RowRecord) -> bool:
            if not rec_fields or set(rec.keys()).issubset(rec_fields):
                return True
            if ignore_errors:
                return False
            raise KeyError(f"Invalid record found {rec}, does not contain the required fields: {rec_fields}")

        new_ids: list[int] = []
        with self.editor, self.insert_cursor(rec_fields) as cur:
            for rec in filter(rec_filter, records):
                new_ids.append(cur.insertRow(tuple(rec.get(k) for k in rec_fields)))
        return tuple(new_ids)

    def delete_identical(self, field_names: Sequence[FieldName] | FieldName) -> dict[int, int]:
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
        with self.editor, self.update_cursor(['OID@'] + list(field_names)) as cur:
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
                
    def filter(self, func: FilterFunc, invert: bool=False) -> Iterator[RowRecord]:
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
    
    def write_attribute_rules(self, out_dir: Path|str) -> None:
        """Write attribute rules out to a structured directory
        
        Note:
            out_dir -> fc_name -> [rule_name.cfg, rule_name.js]
        """
        out_dir = Path(out_dir)
        for rule_name, rule in self.attribute_rules.items():
            _script: str = rule['scriptExpression']
            _cfg = {k:v for k,v in rule.items() if k not in ('scriptExpression',)}
            out_file = out_dir / self.name / rule_name
            out_file.with_suffix('.js').write_text(_script)
            out_file.with_suffix('.cfg').write_text(json.dumps(_cfg, indent=2))
        return
    
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
        name = Path(self.path).relative_to(Path(self.workspace))
        if Exists(copy_fc := Path(workspace) / name):
            raise ValueError(f'{name} already exists in {workspace}!')
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

    def delete_fields(self, fieldnames: Sequence[str]) -> None:
        for fname in fieldnames:
            self.delete_field(fname)
    
    # Magic Methods
    
    if TYPE_CHECKING:
        
        _OVERLOAD_TYPES = (
            FieldName | set[FieldName] | list[FieldName] | tuple[FieldName, ...] | 
            FilterFunc | WhereClause | None
        )
        
        @overload
        def __getitem__(self, field: tuple[FieldName, ...]) -> Iterator[tuple[Any, ...]]:
            """Yield tuples of the requested field values"""
            pass
        
        @overload
        def __getitem__(self, field: list[FieldName]) -> Iterator[list[Any]]:
            """Yield lists of the requested field values"""
            pass
        
        @overload
        def __getitem__(self, field: set[FieldName]) -> Iterator[RowRecord]:
            """Yield dictionaries of the requested field values"""
            pass
  
        @overload
        def __getitem__(self, field: FieldName) -> Iterator[Any]:
            """Yield values from the requested field"""
            pass
        
        @overload
        def __getitem__(self, field: FilterFunc) -> Iterator[RowRecord]:
            """Yield dictionaries of the rows that match the filter function"""
            pass

        @overload
        def __getitem__(self, field: WhereClause) -> Iterator[RowRecord]:
            """Yield values that match the provided WhereClause SQL statement"""
            pass
        
        @overload
        def __getitem__(self, field: None) -> Iterator[None]:
            """Yield nothing (used as fallback if an indexing argument is None)"""
            pass

    def __getitem__(self, field: _OVERLOAD_TYPES) -> Iterator[Any]:
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
                with self.search_cursor(field) as cur:
                    yield from (row for row in cur)
            case list():
                with self.search_cursor(field) as cur:
                    yield from (list(row) for row in cur)
            case set():
                with self.search_cursor(list(field)) as cur:
                    yield from (row for row in as_dict(cur))
            case None:
                yield from () # This allows a side effect None to be used to get nothing

            # Conditional Requests
            case wc if isinstance(wc, WhereClause):
                if not wc.validate(self.fields):
                    raise AttributeError(f'Invalid Where Clause: {wc}, fields not found in {self.name}')
                with self.search_cursor(self.fields, where_clause=wc.where_clause) as cur:
                    yield from (row for row in as_dict(cur))
            case func if callable(func):
                yield from (row for row in self.filter(func))
            case _:
                raise KeyError(
                    f"Invalid option: `{field}` "
                    "Must be a WhereClause, filter functon, field, set of fields, list of fields, or tuple of fields"
                )

    def __contains__(self, field: str) -> bool:
        """Implementation of contains that checks for a field existing in the `FeatureClass`
        """
        return field in self.fields


    def __iter__(self) -> Iterator[dict[str, Any]] | Iterator[Any]:
        """Iterate all rows in the Table or FeatureClass yielding mappings of field name to field value
        
        Note:
            It was decided to yield mappings because without specifying fields, it is up to the user
            to deal with the data as they see fit. Yielding tuples in an order that's not defined by
            the user would be confusing, so a mapping makes it clear exactly what they're accessing
            
        Note:
            When a single field is specified using the `fields_as` context, values will be yielded
        """ 
        with self.search_cursor(self.fields) as cur:
            if len(self.fields) == 1:
                yield from (row for row, in cur)
            else:
                yield from as_dict(cur)

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
        return sum(1 for _ in self['OID@'])

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
            path|pth  : Table or FeatureClass path
            len|length: Table or FeatureClass length (with applied SearchQuery)
            layer|lyr : Linked Table or FeatureClass layer if applicable (else `'None'`)
            shape|shp : Table or FeatureClass shape type
            units|unt : Table or FeatureClass linear unit name
            wkid|code : Table or FeatureClass WKID
            name|nm   : Table or FeatureClass name
            fields|fld: Table or FeatureClass fields (comma seperated)

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
    def where(self, where_clause: str):
        """Apply a where clause to a Table or FeatureClass in a context

        Args:
            where_clause (str): The where clause to apply to the Table or FeatureClass
        
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
            search_options=SearchOptions(where_clause=where_clause)):
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
            mp (Map): The map to add the featureclass to
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
                   ignore_def_query: bool = False,) -> Table:
        """Build a Table or FeatureClass object from a layer applying the layer's current selection to the stored cursors
        
        Args:
            layer (Layer): The layer to convert to a Table or FeatureClass
            ignore_selection (bool): Ignore the layer selection (default: False)
            ignore_definition_query (bool): Ignore the layer definition query (default: False)
        Returns:
            ( Table or FeatureClass ): The Table or FeatureClass object with the layer query applied
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
    
class FeatureClass(Table, Generic[_GeometryType]):
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
        return Describe(self.path) # type: ignore

    @property
    def shape_field_name(self) -> str:
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
        yield from ( shape for shape, in self.search_cursor('SHAPE@'))

    @property
    def spatial_reference(self):
        return self.describe.spatialReference

    @property
    def unit_name(self) -> str:
        return self.spatial_reference.linearUnitName

    @property
    def extent(self) -> Extent:
        """Get the stored extent of the feature class"""
        return self.describe.extent

    @property
    def py_types(self) -> dict[str, type]:
        """Get a mapping of the field types for the FeatureClass"""
        _types = convert_dtypes(self.np_dtypes)
        if 'SHAPE@' in _types and len(self) > 0:
                _types['SHAPE@'] = type(next(self.shapes))
        return _types
    # Data Operations
    
    def footprint(self, buffer: float|None=None) -> _GeometryType | None:
        """Merge all geometry in the featureclass using current SelectionOptions into a single geometry object to use 
        as a spatial filter on other FeatureClasses
        
        Args:
            buffer (float | None): Optional buffer (in feature units, respects projection context) to buffer by (default: None)

        Returns:
            (GeometryType | None): A merged Multi-Geometry of all feature geometries or `None` if no features in FeatureClass
        """
        if len(self) == 0:
            return None

        def merge(acc: _GeometryType, nxt: _GeometryType) -> _GeometryType:
            if buffer:
                nxt = nxt.buffer(buffer)
            return acc.union(nxt)
        
        # Consume the shape generator popping off the first shape and applying the buffer, 
        # Then buffering each additional shape and merging it into the accumulator (starting with _first)
        _shapes = self.shapes
        _first = next(_shapes)
        if buffer:
            _first = _first.buffer(buffer)
        
        return reduce(merge, _shapes, _first)
    
    def recalculate_extent(self) -> None:
        """Recalculate the FeatureClass Extent"""
        RecalculateFeatureClassExtent(self.path, 'STORE_EXTENT')

    # Magic Methods
    
    if TYPE_CHECKING:
        
        _OVERLOAD_TYPES = Table._OVERLOAD_TYPES | Extent | GeometryType
        
        @overload
        def __getitem__(self, field: tuple[FieldName, ...]) -> Iterator[tuple[Any, ...]]:
            """Yield tuples of the requested field values"""
            pass
        
        @overload
        def __getitem__(self, field: list[FieldName]) -> Iterator[list[Any]]:
            """Yield lists of the requested field values"""
            pass
        
        @overload
        def __getitem__(self, field: set[FieldName]) -> Iterator[RowRecord]:
            """Yield dictionaries of the requested field values"""
            pass
  
        @overload
        def __getitem__(self, field: FieldName) -> Iterator[Any]:
            """Yield values from the requested field"""
            pass
        
        @overload
        def __getitem__(self, field: FilterFunc) -> Iterator[RowRecord]:
            """Yield dictionaries of the rows that match the filter function"""
            pass

        @overload
        def __getitem__(self, field: WhereClause) -> Iterator[RowRecord]:
            """Yield values that match the provided WhereClause SQL statement"""
            pass
        
        @overload
        def __getitem__(self, field: None) -> Iterator[None]:
            """Yield nothing (used as fallback if an indexing argument is None)"""
            pass
        
        @overload
        def __getitem__(self, field: GeometryType | Extent) -> Iterator[RowRecord]:
            """Yield rows that intersect the provided geometry"""
            pass

    def __getitem__(self, field: _OVERLOAD_TYPES) -> Iterator[Any]:
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
            case shape if isinstance(shape, Extent | GeometryType):
                with self.search_cursor(self.fields, spatial_filter=shape) as cur:
                    yield from (row for row in as_dict(cur))
            case field if isinstance(field, str|list|tuple|set|Callable|WhereClause):
                yield from super().__getitem__(field)
            case _:
                raise KeyError(
                    f"Invalid option: {field}\n"
                    "Must be a filter functon, set of keys, list of keys, or tuple of keys"
                )
                
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
                   ignore_def_query: bool = False,) -> FeatureClass[_GeometryType]:
        """Build a FeatureClass object from a layer applying the layer's current selection to the stored cursors
        
        Args:
            layer (Layer): The layer to convert to a FeatureClass
            ignore_selection (bool): Ignore the layer selection (default: False)
            ignore_definition_query (bool): Ignore the layer definition query (default: False)
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
    
if __name__ == '__main__':
    pass