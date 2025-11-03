from __future__ import annotations

from pathlib import Path

from collections.abc import (
    Iterator,
)
from typing import (
    Any,
    TypeVar,
)
    
from .featureclass import (
    Table,
    FeatureClass,
)

from arcpy.da import (
    Walk,
    ListDomains,
    Domain,
)

from arcpy.management import (
    DeleteDomain, # pyright: ignore[reportUnknownVariableType]
)

_Default = TypeVar('_Default')
class Dataset:
    """A Container for managing workspace connections.
    
    A Dataset is initialized using `arcpy.da.Walk` and will discover all child datasets, tables, and featureclasses.
    These discovered objects can be accessed by name directly (e.g. `dataset['featureclass_name']`) or by inspecting the
    property of the type they belong to (e.g. dataset.feature_classes['featureclass_name']). The benefit of the second 
    method is that you will be able to know you are getting a `FeatureClass`, `Table`, or `Dataset` object.

    Usage:
        ```python
        >>> dataset = Dataset('dataset/path')
        >>> fc1 = dataset.feature_classes['fc1']
        >>> fc1 = dataset.feature_classes['fc2']
        >>> len(fc1)
        243
        >>> len(fc2)
        778

        >>> count(dataset['fc1'][where('LENGTH > 500')])
        42
        >>> sum(dataset['fc2']['TOTAL'])
        3204903
        ```
    As you can see, the dataset container makes it incredibly easy to interact with data concisely and clearly. 

    Datasets also implement `__contains__` which allows you to check membership from the root node:

    Example:
        ```python
        >>> 'fc1' in dataset
        True
        >>> 'fc6' in dataset
        True
        >>> list(dataset.feature_classes)
        ['fc1', 'fc2']
        >>> list(dataset.datasets)
        ['ds1']
        >>> list(dataset['ds1'].feature_classes)
        ['fc3', 'fc4', 'fc5', 'fc6']
        ```
    """
    def __init__(self, conn: str|Path, *, parent: Dataset|None=None) -> None:
        self.conn = Path(conn)
        self.parent = parent
        self._datasets: dict[str, Dataset] | None = None
        self._feature_classes: dict[str, FeatureClass[Any]] | None=None
        self._tables: dict[str, Table] | None=None
        self.walk()

    @property
    def name(self) -> str:
        return self.conn.stem
    
    @property
    def datasets(self) -> dict[str, Dataset]:
        """A mapping of dataset names to child `Dataset` objects"""
        return self._datasets or {}
    
    @property
    def feature_classes(self) -> dict[str, FeatureClass[Any]]:
        """A mapping of featureclass names to `FeatureClass` objects in the dataset root"""
        return self._feature_classes or {}

    @property
    def tables(self) -> dict[str, Table]:
        """A mapping of table names to `Table` objects in the dataset root"""
        return self._tables or {}

    @property
    def domains(self) -> dict[str, Domain]:
        # Domains only exist in the root dataset
        # Defer to parent until the root is found
        if self.parent:
            return self.parent.domains
        return {d.name: d for d in ListDomains(str(self.conn))}

    @property
    def unused_domains(self) -> dict[str, Domain]:
        return {
            name: domain 
            for name, domain in self.domains.items() 
            if not self.domain_usage(name)
        }

    def domain_usage(self, *domain_names: str) -> list[str]:
        """A list of all Table/FeatureClass names using a specified domain"""
        if not domain_names:
            domain_names = tuple(self.domains.keys())
        
        if invalid := set(domain_names) - set(self.domains):
            raise ValueError(f'{invalid} not found in {self.name} domains')
        
        _features: list[FeatureClass[Any] | Table] = [*self.tables.values(), *self.feature_classes.values()]
        return [
            o.name
            for o in _features
            if any(
                f.domain == domain_name 
                for f in o.describe.fields
                for domain_name in domain_names
            )
        ]

    def delete_domain(self, *domain_names: str) -> None:
        """Delete a domain from the dataset"""
        for domain_name in domain_names:
            DeleteDomain(str(self.conn), domain_name)

    def walk(self) -> None:
        """Traverse the connection/path using `arcpy.da.Walk` and discover all dataset children
        
        Note:
            This is called on dataset initialization and can take some time. Larger datasets can take up to
            a second or more to initialize.
        
        Note:
            If the contents of a dataset change during its lifetime, you may need to call walk again. All 
            children that are already initialized will be skipped and only new children will be initialized
        """
        self._feature_classes = {}
        for root, _, fcs in Walk(str(self.conn), datatype=['FeatureClass']):
            root = Path(root)
            for fc in fcs:
                # Backlink Datasets to parent
                if self.parent is not None and fc in self.parent:
                    self._feature_classes[fc] = self.parent.feature_classes[fc]
                else:
                    self._feature_classes[fc] = FeatureClass(root / fc)
                    
        self._tables = {}
        for root, _, tbls in Walk(str(self.conn), datatype=['Table']):
            root = Path(root)
            self._tables.update({tbl: Table(root / tbl) for tbl in tbls if tbl not in self})
            for tbl in tbls:
                # Backlink Datasets to parent (Should never hit since tables are in root only)
                if self.parent and tbl in self.parent:
                    self._tables[tbl] = self.parent.tables[tbl]
                else:
                    self._tables[tbl] = Table(root / tbl)
        
        # Handle datasets last to allow for backlinking     
        self._datasets = {}
        for root, ds, _ in Walk(str(self.conn), datatype=['FeatureDataset']):
            root = Path(root)
            self._datasets.update({d: Dataset(root / d, parent=self) for d in ds})
    
    def __getitem__(self, key: str) -> FeatureClass[Any] | Table | Dataset:
        # Check top level
        ret = self.tables.get(key) or self.feature_classes.get(key) or self.datasets.get(key)
        if not ret:
            raise KeyError(f'{key} is not a child of {self.conn.stem}')
        return ret
    
    def __contains__(self, key: str) -> bool:
        try:
            self[key]
            return True
        except KeyError:
            return False
        
    def __iter__(self) -> Iterator[Table]:
        for feature_class in self.feature_classes.values():
            yield feature_class
            
        for table in self.tables.values():
            yield table