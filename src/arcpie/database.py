from __future__ import annotations

import json
from pathlib import Path

from tempfile import TemporaryDirectory

from collections.abc import (
    Iterator,
)
from typing import (
    Any,
    Literal,
    TypeVar,
    Unpack,
)

from ._types import (
    domain_param,
    AlterDomainOpts,
    CreateDomainOpts,
)
from arcpie.schemas.workspace import SchemaWorkspace

from .featureclass import (
    Table,
    FeatureClass,
)

from .utils import patch_schema_rules, convert_schema

from arcpy.da import (
    Walk,
    ListDomains,
    Domain,
)

from arcpy.management import (
    DeleteDomain,  # pyright: ignore[reportUnknownVariableType]
    CreateDomain,  # pyright: ignore[reportUnknownVariableType]
    AlterDomain,   # pyright: ignore[reportUnknownVariableType]
    CreateFileGDB, # pyright: ignore[reportUnknownVariableType]
    ConvertSchemaReport, # pyright: ignore[reportUnknownVariableType]
    ImportXMLWorkspaceDocument, # pyright: ignore[reportUnknownVariableType]
    GenerateSchemaReport, # pyright: ignore[reportUnknownVariableType]
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
        
        # Force root dataset to be a gdb, pointing to a folder can cause issues with Walk
        if not parent and self.conn.suffix != '.gdb':
            raise ValueError(f'Root Dataset requires a valid gdb path!')
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
    def domains(self) -> DomainManager:
        return DomainManager(self)

    @property
    def schema(self) -> SchemaWorkspace:
        return json.load(convert_schema(self, 'JSON'))

    def export_rules(self, rule_dir: Path|str) -> None:
        """Export all attribute rules from the dataset into feature subdirectories
        
        Args:
            rule_dir (Path|str): The target directory for the rules
        
        Usage:
            ```python
            >>> # Transfer rules from one dataset to another
            >>> ds.export_rules('my_rules')
            >>> ds2.import_rules('my_rules')
            ```
        """
        for feature_class in self.feature_classes.values():
            feature_class.attribute_rules.export_rules(Path(rule_dir))

    def import_rules(self, rule_dir: Path|str, 
                     *, 
                     skip_fail: bool=False) -> None:
        """Import Attribute rules for the dataset from a directory
        
        Args:
            rule_dir (Path|str): A directory containing rules in feature sub directories
            skip_fail (bool): Skip any attribute rule imports that fail (whole FC) (default: False)
            
        Usage:
            ```python
            >>> # Transfer rules from one dataset to another
            >>> ds.export_rules('my_rules')
            >>> ds2.import_rules('my_rules')
            ```
        """
        rule_dir = Path(rule_dir)
        for feature_class in self.feature_classes.values():
                if not (rule_dir / feature_class.name).exists():
                    continue
                try:
                    feature_class.attribute_rules.import_rules(rule_dir / feature_class.name)
                except Exception as e:
                    if skip_fail:
                        print(f'Failed to import rules for {feature_class.name}: \n\t{e.__notes__}\n\t{e}')
                    else:
                        raise e

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
        if ret := self.tables.get(key) or self.feature_classes.get(key) or self.datasets.get(key):
            return ret
        
        raise KeyError(f'{key} is not a child of {self.conn.stem}')
        
    def get(self, key: str, default: _Default=None) -> FeatureClass[Any] | Table | Dataset | _Default:
        try:
            return self[key]
        except KeyError:
            return default
    
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
            
        for dataset in self.datasets.values():
            yield from dataset
    
    def __len__(self) -> int:
        return sum(1 for _ in self)
           
    def __repr__(self) -> str:
        return (
            "Dataset("
            f"{self.name}, "
            "{"
            f"Features: {len(self.feature_classes)}, "
            f"Tables: {len(self.tables)}, "
            f"Datasets: {len(self.datasets)}"
            "})"
        )
    
    def __str__(self) -> str:
        return self.__fspath__()
    
    def __fspath__(self) -> str:
        return str(self.conn.resolve())

    def export_schema(self, out_loc: Path|str,
                      *,
                      schema_name: str|None=None, 
                      out_format: Literal['JSON', 'XLSX', 'HTML', 'PDF', 'XML']='JSON',
                      remove_rules: bool=False) -> Path:
        """Export the workspace Schema for a GDB dataset
        
        Args:
            out_loc (Path|str): The output location for the workspace schema
            schema_name (str): A name for the schema (default: Dataset.name)
            out_format (Literal['json', 'xml', 'xlsx', 'html']): The output format (default: 'json')
            remove_rules (bool): Don't export associated attribute rules for the dataset (default: False)
            
        Returns:
            Path : The Path object pointing to the output file
        """
        out_loc = Path(out_loc)
        out_loc.mkdir(exist_ok=True, parents=True)
        name = schema_name or self.name
        outfile = (out_loc / name).with_suffix(f'.{out_format.lower()}')
        workspace = json.load(convert_schema(self, out_format))
        schema = patch_schema_rules(workspace, remove_rules=remove_rules)
        with outfile.open('w') as f:
            json.dump(schema, f)
        return outfile
    
    @classmethod
    def from_schema(cls, schema: Path|str, out_loc: Path|str, gdb_name: str, 
                    *,
                    remove_rules: bool=False) -> Dataset:
        """Create a GDB from a schema file (xlsx, json, xml) generated by export_schema
        
        Args:
            schema (Path|str): Path to the schema file
            out_loc (Path|str): Path to the GDB output directory
            gdb_name (str): The name of the gdb
            remove_rules (bool): Don't import Attribute Rules after building the new dataset (default: False)
        
        Usage:
            ```python
            >>> ds = Dataset.from_schema('schema.xlsx', 'out_dir', 'new_db.gdb', skip_rules=True)
            ... # This can take a while depending on the size of the schema
            >>> ds
            Dataset('new_db' {'Features': 10, 'Tables': 3, Datasets: 0})
            ```
        """
        schema = Path(schema)
        out_loc = Path(out_loc)
        new_database = (out_loc / gdb_name).with_suffix('.gdb')

        # Convert the schema to json for easy parsing of attribute rules
        with TemporaryDirectory(f'{gdb_name}_json_schema') as temp:
            temp = Path(temp)
            # Convert the schema to json
            if not schema.suffix == '.json':
                converted_report, = ConvertSchemaReport(
                    str(schema), str(temp), 'json_schema', 'JSON'
                )
            else:
                converted_report = str(schema)
            # Patch the schema doc
            workspace = patch_schema_rules(
                converted_report, remove_rules=remove_rules
            )
            # Write out to tempfile
            patched_schema = temp / 'patched_schema.json'
            patched_schema.write_text(json.dumps(workspace), encoding='utf-8')
            # Convert to importable XML
            xml_schema, = ConvertSchemaReport(
                str(patched_schema), str(temp), 'xml_schema', 'XML'
            )
            # Create a new GDB
            CreateFileGDB(str(out_loc), gdb_name, 'CURRENT')
            # Import the schema doc
            ImportXMLWorkspaceDocument(
                str(new_database), xml_schema, 'SCHEMA_ONLY'
            )
        return Dataset(new_database)

class DomainManager:
    """Handler for interacting with domains defined on a dataset"""
    
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        
    @property
    def dataset(self) -> Dataset:
        """Get the parent Dataset object"""
        return self._dataset
    
    @property
    def workspace(self) -> Path:
        """Get a path to the root workspace that the domains live in"""
        if self.dataset.parent is None:
            return self.dataset.conn
        else:
            return self.dataset.parent.conn
    
    @property
    def domain_map(self) -> dict[str, Domain]:
        """A mapping of domain names to domain objects"""
        if self.dataset.parent:
            return self.dataset.parent.domains.domain_map
        return {d.name: d for d in ListDomains(str(self.dataset.conn))}
    
    @property
    def unused_domains(self) -> dict[str, Domain]:
        return {
            name: domain
            for name, domain in self.domain_map.items() 
            if not self.usage(name)
        }
    
    def __len__(self) -> int:
        return len(self.domain_map)
    
    def __iter__(self) -> Iterator[Domain]:
        """Iterate all domains"""
        yield from self.domain_map.values()
    
    def __getitem__(self, name: str) -> Domain:
        return self.domain_map[name]
    
    def get(self, name: str, default: _Default) -> Domain | _Default:
        try:
            return self[name]
        except KeyError:
            return default
    
    def __contains__(self, domain: str) -> bool:
        return domain in self.domain_map
    
    def usage(self, *domain_names: str) -> dict[str, dict[str, list[str]]]:
        """A mapping of domains to features to fields that shows usage of a domain in a dataset
        
        Args:
            *domain_names (str): Varargs of all domain names to include in the output mapping
        
        Returns:
            ( dict[str, dict[str, list[str]]] ) : A Nested mapping of `Domain Name -> Feature Class -> [Field Name, ...]`
        """
        
        if not domain_names:
            domain_names = tuple(self.domain_map)
        schema = self.dataset.schema
        fc_usage: dict[str, dict[str, list[str]]] = {}
        for ds in schema['datasets']:
            if 'datasets' in ds:
                ds = ds['datasets']
            else:
                ds = [ds]
            for fc in filter(lambda f: 'fields' in f, ds):
                for field in filter(lambda fld: 'domain' in fld, fc['fields']['fieldArray']):
                    assert 'domain' in field
                    if (dn := field['domain']['domainName']) in domain_names:
                        fc_usage.setdefault(dn, {}).setdefault(fc['name'], [])
                        fc_usage[dn][fc['name']].append(field.get('name', '??'))
        return fc_usage
    
    def add_domain(self, domain: Domain|None=None, **opts: Unpack[CreateDomainOpts]) -> None:
        """Add a domain to the parent dataset or the root dataset (gdb)
        
        Args:
            domain (Domain): The domain object to add to the managed Dataset (optional)
            **opts (CreateDomainOpts): Additional overrides that will be applied to the create domain call
        """
        
        if domain:
            args: CreateDomainOpts = {
                'domain_name': domain.name,
                'domain_description': domain.description,
                'domain_type': domain_param(domain.domainType),
                'field_type': domain_param(domain.type),
                'merge_policy': domain_param(domain.mergePolicy),
                'split_policy': domain_param(domain.splitPolicy),
            }
            # Allow overrides
            args.update(opts)
        else:
            args = opts
        CreateDomain(in_workspace=str(self.workspace), **args)
        
    def delete_domain(self, domain: str) -> None:
        """Delete a domain from the workspace
        
        Args:
            domain (str): The name of the domain to delete
        """
        DeleteDomain(str(self.workspace), domain)
    
    def alter_domain(self, domain: str, **opts: Unpack[AlterDomainOpts]) -> None:
        """Alter a domain using the given domain values
        
        Args:
            **opts (AlterDomainOpts): Passthrough for AlterDomain function
        """
        AlterDomain(in_workspace=str(self.workspace), domain_name=domain, **opts)
    