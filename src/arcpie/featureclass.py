"""Sumbodule that helps wrap featureclass objects with useful methods"""

from __future__ import annotations

from arcpy.da import (
    Editor,
    SearchCursor,
    InsertCursor,
    UpdateCursor,
    Subtype,
    ListSubtypes,
    Describe as DescribeDict,
)

from arcpy import (
    Geometry,
    Describe,
)

from arcpy._mp import (
    Layer,
)

import arcpy.typing.describe as dt

from typing import (
    Iterable,
    Any,
    Optional,
    Generator,
    Callable,
)

from typing_extensions import (
    Unpack,
)

from .cursor import (
    SearchOptions, 
    InsertOptions, 
    UpdateOptions,
    ShapeToken,
    EditToken,
    CursorToken,
    CursorTokens,
)

from functools import lru_cache, wraps

FieldName = CursorToken | str

def as_dict(cursor: SearchCursor | UpdateCursor) -> Generator[dict[str, Any], None, None]:
    yield from ( dict(zip(cursor.fields, row)) for row in cursor ) 

class FeatureClass:
    _cache_enabled: bool = False # Set to True to enable caching on all FeatureClass objects

    def __init__(
            self, path: str,
            *,
            search_options: Optional[SearchOptions]=None, 
            update_options: Optional[UpdateOptions]=None, 
            insert_options: Optional[InsertOptions]=None,
            enable_cache: bool=False,
        ) -> None:
        self.path = str(path)
        self._search_options = search_options or SearchOptions()
        self._insert_options = insert_options or InsertOptions()
        self._update_options = update_options or UpdateOptions()
        self._layer: Optional[Layer] = None

    @property
    @lru_cache(maxsize=_cache_enabled)
    def describe(self) -> dt.FeatureClass:
        return Describe(self.path)

    @property
    def name(self) -> str:
        return self.describe.name

    @property
    def search_options(self) -> SearchOptions:
        return self._search_options.copy()
    
    @search_options.setter
    def search_options(self, search_options: SearchOptions) -> None:
        self._search_options = search_options

    @property
    def insert_options(self) -> InsertOptions:
        return self._insert_options.copy()
    
    @insert_options.setter
    def insert_options(self, insert_options: InsertOptions) -> None:
        self._insert_options = insert_options

    @property
    def update_options(self) -> UpdateOptions:
        return self._update_options.copy()
    
    @update_options.setter
    def update_options(self, update_options: UpdateOptions) -> None:
        self._update_options = update_options

    @property
    def fields(self) -> tuple[str, ...]:
        """Tuple of all fieldnames in the FeatureClass"""
        with self.search_cursor('*') as c:
            return c.fields

    @property
    def np_dtypes(self):
        with self.search_cursor('*') as c:
            return c._dtype

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
    def geometry(self, geometry_type: ShapeToken='SHAPE@') -> Generator[Geometry, None, None]:
        yield from ( shape for shape, in self.get_tuples(geometry_type) )

    @property
    @lru_cache(maxsize=_cache_enabled)
    def subtypes(self) -> dict[int, Subtype]:
        """Result of ListSubtypes, mapping of code to Subtype object"""
        return ListSubtypes(self.path)

    @property
    def editor(self) -> Editor:
        return Editor(self.describe.workspace.catalogPath)

    @property
    @lru_cache(maxsize=_cache_enabled)
    def attribute_rules(self):
        return self.describe.fields

    def format_query(self, ids: set[int]) -> str:
        """Format a list of object IDs into a SQL query to be used with cursors or layer selections"""
        return f"{self.describe.OIDFieldName} IN ({','.join(map(str, ids))})"
    
    def _resolve_search_options(self, options: Optional[SearchOptions], overrides: SearchOptions) -> SearchOptions:
        ser_opts = self.search_options
        ser_opts.update(options or  {})
        ser_opts.update(overrides)
        return ser_opts

    def _resolve_insert_options(self, options: Optional[InsertOptions], overrides: InsertOptions) -> InsertOptions:
        ins_opts = self.insert_options
        ins_opts.update(options or {})
        ins_opts.update(overrides)
        return ins_opts

    def _resolve_update_options(self, options: Optional[UpdateOptions], overrides: UpdateOptions) -> UpdateOptions:
        upd_opts = self.update_options
        upd_opts.update(options or {})
        upd_opts.update(overrides)
        return upd_opts

    def search_cursor(self, field_names: FieldName | Iterable[FieldName],
                      *,
                      search_options: Optional[SearchOptions]=None, 
                      **overrides: Unpack[SearchOptions]) -> SearchCursor:
    def insert_cursor(self, field_names: FieldName | Iterable[FieldName],
                      *,
                      insert_options: Optional[InsertOptions], 
                      **overrides: Unpack[InsertOptions]) -> InsertCursor:
        return InsertCursor(self.path, field_names, **self._resolve_insert_options(insert_options, overrides))
    
    def update_cursor(self, field_names: FieldName | Iterable[FieldName],
                      *,
                      update_options: Optional[UpdateOptions], 
                      **overrides: Unpack[UpdateOptions]) -> UpdateCursor:
        return UpdateCursor(self.path, field_names, **self._resolve_update_options(update_options, overrides))
    
    def get_records(self, field_names: Iterable[FieldName], **options: Unpack[SearchOptions]):
        """Generate row dicts with in the form `{field: value, ...}` for each row in the cursor

        Parameters:
            field_names (str | Iterable[str]): The columns to iterate

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

    @classmethod
    def from_layer(cls, layer: Layer) -> FeatureClass:
        """Build a FeatureClass object from a layer applying the layer's current selection to the stored cursors
        
        Parameters:
            layer (Layer): The layer to convert to a FeatureClass
        
        Returns:
            ( FeatureClass ): The FeatureClass object with the layer query applied
        """
        selected_ids: set[int] = layer.getSelectionSet() # type: ignore (this function always returns set[int])
        fc = cls(layer.dataSource)
        search_options = SearchOptions(where_clause=fc.format_query(selected_ids))
        update_options = UpdateOptions(where_clause=fc.format_query(selected_ids))
        fc.search_options = search_options
        fc.update_options = update_options
        return fc
    

if __name__ == '__main__':
    f = FeatureClass('path')