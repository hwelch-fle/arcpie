"""Sumbodule that helps wrap featureclass objects with useful methods"""

from __future__ import annotations

# Typing imports
import arcpy.typing.describe as dt

from typing import (
    Iterable,
    Any,
    Optional,
    Generator,
    Callable,
    TypeVar,
    TypeAlias,
    Generic,
    Literal,
    TYPE_CHECKING,
    overload,
)

# Arcpy imports
from arcpy.da import (
    Editor,
    SearchCursor,
    InsertCursor,
    UpdateCursor,
    ListSubtypes,
)

if TYPE_CHECKING:
    from arcpy.da import (
        SpatialRelationship,
        SearchOrder,
        Subtype,
    )
else:
    SpatialRelationship = None
    SearchOrder = None
    Subtype = None

from arcpy import (
    Geometry,
    Polygon,
    Polyline,
    PointGeometry,
    Multipoint,
    Extent,
    Describe,
    SpatialReference,
    Exists,
)

from arcpy.management import (
    CopyFeatures,    
)

from arcpy._mp import (
    Layer,
    Map,
)

from typing_extensions import (
    Unpack,
)

# Standardlib imports
from contextlib import (
    contextmanager,
)

import json

from pathlib import (
    Path,
)

# Library imports
from cursor import (
    SearchOptions, 
    InsertOptions, 
    UpdateOptions,
    ShapeToken,
    EditToken,
    CursorToken,
    CursorTokens,
    SQLClause,
    _Geometry,
)

FieldName = CursorToken | str

def as_dict(cursor: SearchCursor | UpdateCursor) -> Generator[dict[str, Any], None, None]:
    yield from ( dict(zip(cursor.fields, row)) for row in cursor ) 

def format_query(vals: Iterable[Any]) -> str:
    """Format a list of values into a SQL list"""
    return f"({','.join(map(str, vals))})"

_Geo_T = TypeVar('_Geo_T', Geometry, Polygon, PointGeometry, Polyline, Multipoint)
class FeatureClass(Generic[_Geo_T]):
    """A Wrapper for ArcGIS FeatureClass objects
    
    Example:
        ```python
        >>> point_features = FeatureClass[PointGeometry]('<feature_class_path>') # Initialize FeatureClass with Geometry Type
        >>> buffers = (pt.buffer(10) for pt in point_features.shapes)            # Create a buffer Generator
        >>> 
        >>> sr = SpatialReference(4206)
        >>> with point_features.reference_as(sr):                                # Set a new spatial reference
        >>>     for buffer in buffers:                                           # Consume the Generator, but with the new reference
        >>>         area = buffer.area
        >>>         units = sr.linearUnitName
        >>>         print(f"{area} Sq{units}")
        >>>
    """

    def __init__(
            self, path: str,
            *,
            search_options: Optional[SearchOptions]=None, 
            update_options: Optional[UpdateOptions]=None, 
            insert_options: Optional[InsertOptions]=None,
            clause: Optional[SQLClause]=None,
            shape_token: ShapeToken='SHAPE@'
        ) -> None:
        self.path = str(path)
        self._clause = clause or SQLClause(None, None)
        self._search_options = search_options or SearchOptions()
        self._insert_options = insert_options or InsertOptions()
        self._update_options = update_options or UpdateOptions()

        self._shape_token: ShapeToken = shape_token
        self._layer: Optional[Layer] = None
        self._in_edit_session=False

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
        return self._update_options.copy()
    
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
    def layer(self) -> Optional[Layer]:
        return self._layer

    @layer.setter
    def layer(self, layer: Layer) -> None:
        """Set a layer object for the FeatureClass, layer datasource must be this feature class!"""
        if layer.dataSource != self.path:
            raise ValueError(f'Layer: {layer.name} does not source to {self.name} FeatureClass at {self.path}!')
        self._layer = layer

    @property
    def shape_token(self) -> ShapeToken:
        return self._shape_token

    @shape_token.setter
    def shape_token(self, shape_token: ShapeToken) -> None:
        self._shape_token = shape_token

    # ro Properties
    @property
    def describe(self) -> dt.FeatureClass:
        return Describe(self.path)

    @property
    def workspace(self) -> str:
        """Get the workspace of the `FeatureClass`"""
        return self.describe.workspace.catalogPath

    @property
    def name(self) -> str:
        return self.describe.name

    @property
    def shape_field_name(self) -> str:
        return self.describe.shapeFieldName

    @property
    def oid_field_name(self) -> str:
        return self.describe.OIDFieldName

    @property
    def fields(self) -> tuple[FieldName, ...]:
        """Tuple of all fieldnames in the FeatureClass"""
        exclude = (self.oid_field_name, self.shape_field_name)
        replace = ('OID@', self.shape_token)
        with self.search_cursor('*') as c:
            return replace + tuple((f for f in c.fields if f not in exclude))

    @property
    def np_dtypes(self):
        with self.search_cursor('*') as c:
            return c._dtype

    @property
    def subtypes(self) -> dict[int, Subtype]:
        """Result of ListSubtypes, mapping of code to Subtype object"""
        return ListSubtypes(self.path)

    @property
    def shapes(self) -> Generator[_Geo_T, None, None]:
        yield from (shape for shape, in self.search_cursor('SHAPE@'))

    @property
    def spatial_reference(self):
        return self.describe.spatialReference

    @property
    def unit_name(self):
        return self.spatial_reference.linearUnitName

    @property
    def is_editing(self) -> bool:
        """Returns true if the featureclass is currently within an edit session"""
        return self._in_edit_session    

    @property
    def extent(self) -> Extent:
        """Get the stored extent of the feature class"""
        return self.describe.extent

    # Option Resolvers (kwargs -> Options Object -> FeatureClass Options)
    def _resolve_search_options(self, options: Optional[SearchOptions], overrides: SearchOptions) -> SearchOptions:
        """Combine all provided SearchOptions into one dictionary"""
        return {'sql_clause': self.clause or SQLClause(None, None), **self.search_options, **(options or {}), **overrides}

    def _resolve_insert_options(self, options: Optional[InsertOptions], overrides: InsertOptions) -> InsertOptions:
        """Combine all provided InsertOptions into one dictionary"""
        return {**self.insert_options, **(options or {}), **overrides}

    def _resolve_update_options(self, options: Optional[UpdateOptions], overrides: UpdateOptions) -> UpdateOptions:
        """Combine all provided UpdateOptions into one dictionary"""
        return {'sql_clause': self.clause or SQLClause(None, None), **self.update_options, **(options or {}), **overrides}

    # Cursor Handlers
    def search_cursor(self, field_names: FieldName | Iterable[FieldName],
                      *,
                      search_options: Optional[SearchOptions]=None, 
                      **overrides: Unpack[SearchOptions]) -> SearchCursor:
        """Get a `SearchCursor` for the `FeatureClass`
        Supplied search options are resolved by updating the base FeatureClass Search options in this order:

        `**overrides['kwarg'] -> search_options['kwarg'] -> self.search_options['kwarg']`

        This is implemented using unpacking operations with the lowest importance option set being unpacked first

        `{**self.search_options, **(search_options or {}), **overrides}`
        
        With direct key word arguments (`**overrides`) shadowing all other supplied options. This allows a Feature Class to
        be initialized using a base set of options, then a shared SearchOptions set to be applied in some contexts,
        then a direct keyword override to be supplied while never mutating the base options of the feature class.
        
        Arguments:
            field_names (str | Iterable[str]): The column names to include from the `FeatureClass`
            search_options (Optional[SearchOptions]): A `SeachOptions` instance that will be used to shadow
                `search_options` set on the `FeatureClass`
            **overrides ( Unpack[SeachOptions] ): Additional keyword arguments for the cursor that shadow 
                both the `seach_options` variable and the `FeatureClass` instance `SearchOptions`
        
        Returns:
            ( SearchCursor ): A `SearchCursor` for the `FeatureClass` instance that has all supplied options
                resolved and applied
                
        Example:
            ```python
                >>> cleese_search = SearchOptions(where_clause="NAME = 'John Cleese'")
                >>> idle_search = SearchOptions(where_clause="NAME = 'Eric Idle'")
                >>> monty = FeatureClass('<path>', search_options=cleese_search)
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

    def insert_cursor(self, field_names: FieldName | Iterable[FieldName],
                      *,
                      insert_options: Optional[InsertOptions]=None, 
                      **overrides: Unpack[InsertOptions]) -> InsertCursor:
        """See `FeatureClass.search_cursor` doc for general info. Operation of this method is identical but returns an `InsertCursor`"""
        return InsertCursor(self.path, field_names, **self._resolve_insert_options(insert_options, overrides))

    def update_cursor(self, field_names: FieldName | Iterable[FieldName],
                      *,
                      update_options: Optional[UpdateOptions]=None, 
                      **overrides: Unpack[UpdateOptions]) -> UpdateCursor:
        """See `FeatureClass.search_cursor` doc for general info. Operation of this method is identical but returns an `UpdateCursor`"""
        return UpdateCursor(self.path, field_names, **self._resolve_update_options(update_options, overrides))

    def distinct(self, distinct_fields: Iterable[FieldName] | FieldName) -> Generator[tuple[Any, ...]]:
        """Yield rows of distinct values
        
        Arguments:
            distinct_fields (Iterable[FieldName] | FieldName): The field or fields to find distinct values for.
                Choosing multiple fields will find all distinct instances of those field combinations
        
        Yields:
            ( tuple[Any, ...] ): A tuple containing the distinct values (single fields will yield `(value, )` tuples)
        """
        clause = SQLClause(prefix=f'DISTINCT {format_query(distinct_fields)}', postfix=None)
        yield from ( value for value in self.search_cursor(distinct_fields, sql_clause=clause) )

    def get_records(self, field_names: Iterable[FieldName], **options: Unpack[SearchOptions]):
        """Generate row dicts with in the form `{field: value, ...}` for each row in the cursor

        Parameters:
            field_names (str | Iterable[str]): The columns to iterate
            search_options (SearchOptions): A Search Options object
            **options (Unpack[SearchOptions]): Additional over
            search_options (SearchOptions): A Search Options object
            **options (Unpack[SearchOptions]): Additional over
        Yields 
            ( dict[str, Any] ): A mapping of fieldnames to field values for each row
        """
        yield from as_dict(self.search_cursor(field_names, **options))

    def get_tuples(self, field_names: Iterable[FieldName], **options: Unpack[SearchOptions]) -> Generator[tuple[Any, ...]]:
        """Generate tuple rows in the for (val1, val2, ...) for each row in the cursor
        
        Parameters:
            field_names (str | Iterable[str]): The columns to iterate
            **options (SearchOptions): Additional parameters to pass to the SearchCursor
        """
        yield from self.search_cursor(field_names, **options)

    def insert_records(self, records: Iterable[dict[str, Any]], ignore_errors: bool=False) -> tuple[int]:
        """Provide a list of records to insert
        Args:
            records (Iterable[dict[str, Any]]): The sequence of records to insert
            ignore_errors (bool): Ignore per-row errors and continue. Otherwise raise KeyError (default: True)
        
        Returns:
            ( tuple[int] ): Returns the OIDs of the newly inserted rows

        Raises:
            ( KeyError ): If the records have varying keys or the keys are not in the FeatureClass
            
        Usage:
            ```python
            >>> new_rows = [
            ...    {'first': 'John', 'last': 'Cleese', 'year': 1939}, 
            ...    {'first': 'Michael', 'last': 'Palin', 'year': 1943}
            ... ]
            >>> print(fc.insert_rows(new_rows))
            ... (2,3)
            ...
            >>> # Insert all shapes from fc into fc2
            >>> fc2.insert_rows(fc.get_records(['first', 'last', 'year']))
            ... (1,2)
            ```
        """
        # Grab the first record
        # Doing it this way allows generators to be passed
        _first_rec = None
        for record in records:
            _first_rec = record
            break

        # Nothing to insert
        if not _first_rec:
            return tuple()

        # Confirm that the first record has valid field names
        rec_fields = sorted(_first_rec.keys())
        if set(rec_fields) != set(*self.fields, *CursorTokens):
            raise KeyError(f"Provided Record is not a valid subset of {self.name} fields:\n{self.fields}")

        # Create a key filter to remove any invalid records or raise a KeyError
        def rec_filter(rec: dict) -> bool:
            _valid = rec.keys() == set(rec_fields)
            if _valid:
                return True
            if ignore_errors:
                return False
            raise KeyError(f"Invalid record found {rec}, does not contain the required fields: {rec_fields}")

        with self.editor(), self.insert_cursor(rec_fields) as cur:
            new_ids = []
            # Handle case where records is a generator and the field validation 
            # consumed the first record
            if isinstance(records, Generator):
                new_ids.append(cur.insertRow([_first_rec.get(k) for k in rec_fields]))
            for rec in filter(rec_filter, records):
                new_ids.append(cur.insertRow([rec.get(k) for k in rec_fields]))
            return tuple(new_ids)

    def filter(self, func: Callable[[dict[FieldName, Any]], bool], invert: bool=False) -> Generator[dict[FieldName, Any]]:
        """Apply a function filter to rows in the FeatureClass

        Args:
            func (Callable[[dict[str, Any]], bool]): A callable that takes a 
                row dictionary and returns True or False
            invert (bool): Invert the function. Only yield rows that return `False`
        
        Yields:
            ( dict[str, Any] ): Rows in the FeatureClass that match the filter (or inverted filter)

        Usage:
            ```python
            >>> def area_filter(row: dict) -> bool:
            >>>     return row['Area'] >= 10

            >>> for row in fc:
            >>>     print(row['Area'])
            ... 1
            ... 2
            ... 10
            ... ...
            
            >>> for row in fc.filter(area_filter):
            >>>     print(row['Area'])
            ... 10
            ... 11
            ... 90
            ... ...
            ```

        """
        yield from ( row for row in self if func(row) == (not invert) )

    # Data Operations
    def copy(self, workspace: str, options: bool=True) -> FeatureClass:
        """Copy this `FeatureClass` to a new workspace
        
        Arguments:
            workspace (str): The path to the workspace
            options (bool): Copy the cursor options to the new `FeatureClass` (default: `True`)
            
        Returns:
            (FeatureClass): A `FeatureClass` instance of the copied features
        
        Example:
            ```python
            >>> new_fc = fc.copy('workspace2')
            >>> new_fc == fc
            False
        """
        name = Path(self.path).relative_to(Path(self.workspace))
        if Exists(copy_fc := Path(workspace) / name):
            raise ValueError(f'{name} already exists in {workspace}!')
        CopyFeatures(self.path, str(copy_fc))
        fc = FeatureClass(str(copy_fc))
        if options:
            fc.search_options = self.search_options
            fc.update_options = self.update_options
            fc.insert_options = self.insert_options
            fc.clause = self.clause
        return fc

    def clear(self, all: bool=False) -> int:
        """Delete all rows in the `FeatureClass` that are returned with the active `update_options`

        Arguments:
            all (bool): Set to `True` to clear all rows ignoring supplied `update_options`

        Returns:
            (int): The number of rows deleted
            
        Note:
            With `all` not set to `True`, only rows that match the `update_options` settings will be deleted
        
        Warning:
            No way to undo this!
        """
        with self.update_cursor('OID@') as cur:
            return sum(cur.deleteRow() or 1 for _ in cur)

    def esrijson(self, display_field: Optional[FieldName]=None) -> str:
        """Dump the current state of the `FeatureClass` to an esrijson string"""
        return json.dumps(
            {
                'displayFieldName': display_field,
                'fieldAliases': {
                    f.name : f.aliasName
                    for f in self.describe.fields
                    if f.aliasName
                },
                'geometryType': self.describe.shapeType,
                'hasZ': self.describe.hasZ,
                'hasM': self.describe.hasM,
                'spatialReference' : self.spatial_reference.exportToString(),
                'fields' : [
                    {
                        'name': f.name,
                        'type': f.type,
                        'alias': f.aliasName,
                    }
                    for f in self.describe.fields
                ],
                'features': [
                    {
                        'geometry': row.pop('SHAPE@'),
                        'attributes': row
                    }
                    for row in self.get_records(['SHAPE@'] + list(self.fields))
                ]
            }
        )
    
    def geojson(self) -> str:
        return json.dumps(
            {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'id': row['OID@'],
                        'geometry': {
                            row.pop('SHAPE@').JSON
                        },
                        'properties': row
                    }
                    for row in self.get_records(['SHAPE@', 'OID@'] + list(self.fields))
                ]
            }
        )

    # Magic Methods
    if TYPE_CHECKING:
        
        @overload
        def __getitem__(self, field: tuple[FieldName, ...]) -> Generator[tuple[Any, ...]]:
            pass
        
        @overload
        def __getitem__(self, field: list[FieldName]) -> Generator[list[Any]]:
            pass
        
        @overload
        def __getitem__(self, field: set[FieldName]) -> Generator[dict[FieldName, Any]]:
            pass
  
        @overload
        def __getitem__(self, field: Callable[[dict[FieldName, Any]], bool]) -> Generator[dict[FieldName, Any]]:
          pass

        @overload
        def __getitem__(self, field: FieldName) -> Generator[Any, None, None]:
          pass


    def __getitem__(self, field) -> Generator[Any]:
        """Create a generator that yields single values from the requested column"""
        match field:
            case str():
                yield from ( val for val, in self.search_cursor(field) )
            case tuple():
                yield from ( row for row in self.search_cursor(field) )
            case list():
                yield from ( list(row) for row in self.search_cursor(field) )
            case set():
                yield from ( row for row in as_dict(self.search_cursor(field)) )
            case Callable():
                yield from ( row for row in self.filter(field) )
            case _:
                raise KeyError(
                    f"Invalid option: {field}\n"
                    "Must be a filter functon, set of keys, list of keys, or tuple of keys"
                )

    def __iter__(self) -> Generator[dict[str, Any]]:
        """Iterate all rows in the FeatureClass yielding mappings of field name to field value"""
        yield from ( as_dict(self.search_cursor(self.fields)) )

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
        """Provide a constructor string e.g. `FeatureClass[Polygon]('path')`"""
        return f"{self.__class__.__name__}[{_Geo_T.__name__}]('{self.path}')"

    def __str__(self) -> str:
        """Return the `FeatureClass` path for use with other arcpy methods"""
        return self.path

    def __eq__(self, other) -> bool:
        """Determine if the datasource of two featureclass objects is the same"""
        return isinstance(other, self.__class__) and self.path == other.path

    def __format__(self, format_spec: str) -> str:
        """Implement format specs for string formatting a featureclass.

        Warning:
            The `{fc:len}` spec should only be used when needed. This spec will call `__len__` when 
            used and will traverse the entire FeatureClass with applied SearchOptions each time it is 
            called. See: `__len__` doc for info on better ways to track counts in loops.

        Arguments:
            path|pth  : FeatureClass path
            len|length: FeatureClass length (with applied SearchQuery)
            layer|lyr : Linked FeatureClass layer if applicable (else `'None'`)
            shape|shp : FeatureClass shape type
            units|unt : FeatureClass linear unit name
            wkid      : FeatureClass WKID
            name|nm   : FeatureClass name
            fields|fld: FeatureClass fields (comma seperated)
        Usage:
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
            case 'shape' | 'shp':
                return self.describe.shapeType
            case 'units' | 'unt':
                return self.unit_name
            case 'wkid':
                return str(self.spatial_reference.factoryCode)
            case 'name' | 'nm':
                return self.name
            case 'fields' | 'flds':
                return ','.join(self.fields)
            case _:
                return str(self)

    # Context Managers
    @contextmanager
    def editor(self, multiuser_mode: Optional[bool]=True):
        """Create an editor context for the feature, required for features that participate in Topologies or exist
        on remote servers
        
        Arguments:
            multiuser_mode (bool): When edits will be performed on versioned data, set this to `True`; otherwise, set it to `False`. 
                Only use with enterprise geodatabases. (default: `True`)
        
        Yields:
            (self): Yields the featureclass back to you within an edit context and with the `is_editing` flag set

        Usage:
            ```python
            new_rows = [('John', 'Cleese', 1939), ('Michael', 'Palin', 1943)]
            
            new_ids = []
            with fc.editor:
                with fc.insert_cursor(['first', 'last', 'year']) as cur:
                    for r in new_rows:
                        new_ids.append(cur.insertRow(r))

            # --OR-- (This is a much cleaner way)

            with fc.editor, fc.insert_cursor(['first', 'last', 'year']) as cur:
                new_ids = [cur.insertRow(r) for r in new_rows]
            ```
        """
        with Editor(self.path, multiuser_mode=multiuser_mode):
            try:
                self._in_edit_session = True
                yield self
            finally:
                self._in_edit_session = False

    @contextmanager
    def reference_as(self, spatial_reference: SpatialReference):
        """Allows you to temporarily set a spatial reference on SearchCursor and UpdateCursor objects within a context block
        
        Arguments:
            spatial_reference (SpatialReference): The spatial reference to apply to the cursor objects
        
        Yields:
            (self): Mutated self with search and update options set to use the provided spatial reference

        Examples:
            ```python
            >>> sr = arcpy.SpatialReference(26971)
            >>> fc = FeatureClass[Polygon]('<fc_path>')
               
            >>> orig_shapes = list(fc.shapes)
               
            >>> with fc.project_as(sr):
            >>>     proj_shapes = list(fc.shapes)
               
            >>> print(orig_shapes[0].spatialReference)
            SpatialReference(4326)
            
            >>> print(proj_shapes[0].spatialReference)
            SpatialReference(26971)
            ```
        """
        _old_src_ref = self.search_options.get('spatial_reference')
        _old_upd_ref = self.update_options.get('spatial_reference')

        try:
            self._search_options['spatial_reference'] = spatial_reference
            self._update_options['spatial_reference'] = spatial_reference
            yield self

        finally:
            if _old_src_ref:
                self._search_options['spatial_reference'] = _old_src_ref
            if _old_upd_ref:
                self._update_options['spatial_reference'] = _old_upd_ref

    @contextmanager
    def options(self,
                *, 
                strict: bool = False,
                search_options: Optional[SearchOptions]=None, 
                update_options: Optional[UpdateOptions]=None, 
                insert_options: Optional[InsertOptions]=None, 
                clause: Optional[SQLClause]=None):
        """Enter a context block where the supplied options replace the stored options for the `FeatureClass`
        
        Arguments:
            strict (bool): If this is set to `True` the `FeatureClass` will not fallback on existing options
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
            self._search_options = search_options or _src_ops if not strict else SearchOptions()
            self._update_options = update_options or _upd_ops if not strict else UpdateOptions()
            self._insert_options = insert_options or _ins_ops if not strict else InsertOptions()
            self._clause = clause or _clause if not strict else SQLClause(None, None)
            yield self

        finally:
            self._search_options = _src_ops
            self._update_options = _upd_ops
            self.insert_options = _ins_ops
            self._clause = _clause

    @contextmanager
    def where(self, where_clause: str):
        """Apply a where clause to a FeatureClass in a context

        Args:
            where_clause (str): The where clause to apply to the FeatureClass
        
        Usage:
            ```python
            >>> with fc.where("first = 'John'") as f:
            >>>     for f in fc:
            >>>         print(f)
            ... {'first': 'John', 'last': 'Cleese', 'year': 1939}

            >>> with fc.where('year > 1939'):
            >>>     print(len(fc))
            ... 5
            >>> print(len(fc))
            ... 6
            ```

        Note:
            This method of filtering a FeatureClass will always be more performant than using the 
            `.filter` method. If you can achieve the filtering you want with a where clause, do it.
        """
        with self.options(search_options=SearchOptions(where_clause=where_clause)):
            yield self

    @contextmanager
    def spatial_filter(self, spatial_filter: _Geometry | Extent, spatial_relationship: SpatialRelationship='INTERSECTS'):
        """Apply a spatial filter to the FeatureClass in a context
        
        Args:
            spatial_filter (Geometry | Extent): The geometry to use as a spatial filter
            spatial_relationship (SpatialRelationship): The relationship to check for (default: `INTERSECTS`)
        
        Usage:
            ```python
            >>> with fc.spatial_filter(boundary) as f:
            >>>     print(len(fc))
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
                
        """
        with self.options(search_options=SearchOptions(spatial_filter=spatial_filter, spatial_relationship=spatial_relationship)):
            yield self

    # Mapping interfaces (These pass common `Layer` operations up to the FeatureClass)
    def bind_to_layer(self, layer: Layer) -> None:
        """Update the provided layer's datasource to this FeatureClass
        
        Args:
            layer (Layer): The layer to update connection properties for
        """
        layer.updateConnectionProperties(layer.dataSource, self.path)

    def add_to_map(self, map: Map, pos: Literal['AUTO_ARRANGE', 'BOTTOM', 'TOP']='AUTO_ARRANGE') -> None:
        """Add the featureclass to a map

        Note: 
            If the FeatureClass has a layer, the bound layer will be added to the map. 
            Otherwise a default layer will be added. And the new layer will be bound to the FeatureClass

        Args:
            mp (Map): The map to add the featureclass to
        """
        if not self.layer:
            # Create a default layer, bind it, remove, and add back
            # with addLayer to match behavior with existing bound layer
            self.layer = map.addDataFromPath(self.path) #type:ignore (Always Layer)
            map.removeLayer(self.layer)
        map.addLayer(self.layer, pos)

    def select(self, method: Literal['NEW','DIFFERENCE','INTERSECT','SYMDIFFERENCE','UNION']='NEW') -> set[int]:
        """If the FeatureClass is bound to a layer, update the layer selection with the active SearchOptions
        
        Args:
            method: The method to use to apply the selection\n
                `DIFFERENCE`: Selects the features that are not in the current selection but are in the FeatureClass.\n
                `INTERSECT`: Selects the features that are in the current selection and the FeatureClass.\n
                `NEW`: Creates a new feature selection from the FeatureClass.\n
                `SYMDIFFERENCE`: Selects the features that are in the current selection or the FeatureClass but not both.\n
                `UNION`: Selects all the features in both the current selection and those in FeatureClass.\n
        
        Returns:
            set[int] The selected OIDs
        """
        if not self.layer:
            return set()
        
        self.layer.setSelectionSet(list(self['OID@']), method=method) 
        return self.layer.getSelectionSet() #type:ignore (Always set[int])
        
    def unselect(self) -> set[int]:
        """Remove all layer selections
        
        Returns:
            set[int] The selection that is being removed
        """
        if not self.layer:
            return set()
        try:
            return self.layer.getSelectionSet() #type:ignore (Always set[int])
        finally:
            self.layer.setSelectionSet(method='NEW')

    # Factory Constructors
    @classmethod
    def from_layer(cls, layer: Layer, 
                   *,
                   max_selection: int=500_000, # This needs testing
                   raise_exception: bool=False) -> FeatureClass:
        """Build a FeatureClass object from a layer applying the layer's current selection to the stored cursors
        
        Parameters:
            layer (Layer): The layer to convert to a FeatureClass
            max_selection (int): Maximum number of records allowed in the selection
                use this to prevent a SQL query with millions of OIDs from being generated
            raise_exception (bool): If this flag is set, a `max_selection` overrun will raise a `ValueError`
                otherwise, it will print a warning to `stdout` and continue
            max_selection (int): Maximum number of records allowed in the selection
                use this to prevent a SQL query with millions of OIDs from being generated
            raise_exception (bool): If this flag is set, a `max_selection` overrun will raise a `ValueError`
                otherwise, it will print a warning to `stdout` and continue
        Returns:
            ( FeatureClass ): The FeatureClass object with the layer query applied
        
        Raises:
            ( ValueError ): If the layer selection set is greater than the `max_selection` arg and the `raise_exception` flag is set
        
        Raises:
            ( ValueError ): If the layer selection set is greater than the `max_selection` arg and the `raise_exception` flag is set
        """
        fc = cls(layer.dataSource)
        selected_ids: set[int] = layer.getSelectionSet() # type: ignore (this function always returns set[int])

        if len(selected_ids) > max_selection:
            selected_ids = set()
            if raise_exception:
                raise ValueError(f'Layer has a selection set of {len(selected_ids)}, '
                                 f'which is greater that the max limit of {max_selection}')
            print(f'Layer: {layer.name} selection exceeds maximum, removed selection for {fc.name}')

        selected = f"{fc.describe.OIDFieldName} IN {format_query(selected_ids)}"
        fc.search_options = SearchOptions(where_clause=selected)
        fc.update_options = UpdateOptions(where_clause=selected)
        return fc
    

if __name__ == '__main__':
    fc = FeatureClass[Polygon]('path')