from __future__ import annotations

from pathlib import Path

from .featureclass import (
    Table,
    FeatureClass,
    GeometryType,
)

from arcpy.da import (
    Walk,
)

class Dataset:
    def __init__(self, conn: str|Path) -> None:
        self.conn = Path(conn)
        self._datasets: dict[str, Dataset] | None = None
        self._feature_classes: dict[str, FeatureClass[GeometryType]] | None=None
        self._tables: dict[str, Table] | None=None
        self.walk()

    @property
    def datasets(self) -> dict[str, Dataset]:
        return self._datasets or {}
    
    @property
    def feature_classes(self) -> dict[str, FeatureClass[GeometryType]]:
        return self._feature_classes or {}

    @property
    def tables(self) -> dict[str, Table]:
        return self._tables or {}

    def walk(self) -> None:
        self._feature_classes = {}
        for root, ds, fcs in Walk(str(self.conn), datatype=['FeatureClass']):
            root = Path(root)
            if ds:
                self._datasets = self._datasets or {}
                self._datasets.update({d: Dataset(root / d) for d in ds if d not in self})
            else:
                self._feature_classes.update({fc: FeatureClass(root / fc) for fc in fcs if fc not in self})
        self._tables = {}
        for root, ds, tbls in Walk(str(self.conn), datatype=['Table']):
            root = Path(root)
            self._tables.update({tbl: Table(root / tbl) for tbl in tbls if tbl not in self})
    
    def __getitem__(self, key: str) -> FeatureClass[GeometryType] | Table | Dataset:
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