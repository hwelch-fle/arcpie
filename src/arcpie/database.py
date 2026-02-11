from __future__ import annotations

from functools import cached_property
import json
from pathlib import Path

from tempfile import TemporaryDirectory

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    Unpack,
    overload,
)

from arcpy import Describe # pyright: ignore[reportUnknownVariableType]

from arcpie.cursor import Field

from ._types import (
    AttributeRule,
    domain_param,
    AlterDomainOpts,
    CreateDomainOpts,
    RelationshipOpts,
    RelationshipRemoveRuleOpts,
    RelationshipAddRuleOpts,
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
    DeleteDomain,  # type: ignore (incorrect hinting from arcpy)
    CreateDomain,  # type: ignore (incorrect hinting from arcpy)
    AlterDomain,   # type: ignore (incorrect hinting from arcpy)
    CreateFileGDB, # type: ignore (incorrect hinting from arcpy)
    ConvertSchemaReport, # type: ignore (incorrect hinting from arcpy)
    ImportXMLWorkspaceDocument, # type: ignore (incorrect hinting from arcpy)
    Delete, # type: ignore (incorrect hinting from arcpy)
    CreateRelationshipClass, # type: ignore (incorrect hinting from arcpy)
    AddRuleToRelationshipClass, # type: ignore (incorrect hinting from arcpy)
    RemoveRuleFromRelationshipClass, # type: ignore (incorrect hinting from arcpy)
    CreateFeatureclass, # type: ignore (incorrect hinting from arcpy)
    CreateTable, # type: ignore (incorrect hinting from arcpy)
    CreateFeatureDataset, # type: ignore (incorrect hinting from arcpy)
)

if TYPE_CHECKING:
    from arcpy.typing.describe import RelationshipClass
else:
    RelationshipClass = object

_Default = TypeVar('_Default')
if TYPE_CHECKING: # 3.13 features (default for TypeVar)
    _Schema = TypeVar('_Schema', default=Mapping[str, Any])
else:
    _Schema = TypeVar('_Schema')
    
class Dataset(Generic[_Schema]):
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
    def __init__(self, conn: str|Path, *, parent: Dataset[Any]|None=None) -> None:
        self.conn = Path(conn)
        
        # Force root dataset to be a gdb, pointing to a folder can cause issues with Walk
        if not parent and self.conn.suffix != '.gdb':
            raise ValueError('Root Dataset requires a valid gdb path!')
        self.parent = parent
        self._datasets: dict[str, Dataset[Any]] | None = None
        self._feature_classes: dict[str, FeatureClass] | None = None
        self._tables: dict[str, Table[Any]] | None = None
        self._relationships: dict[str, Relationship]
        self._annotation_features: dict[str, FeatureClass] | None = None
        self.walk()

    @property
    def name(self) -> str:
        return self.conn.stem
    
    @property
    def datasets(self) -> dict[str, Dataset[Any]]:
        """A mapping of dataset names to child `Dataset` objects"""
        return self._datasets or {}
    
    @property
    def feature_classes(self) -> dict[str, FeatureClass]:
        """A mapping of featureclass names to `FeatureClass` objects in the dataset root"""
        return self._feature_classes or {}

    @property
    def annotations(self) -> dict[str, FeatureClass]:
        """A mapping of annotation names to `FeatureClass` objects"""
        return {k: v for k, v in self.feature_classes.items() if v.describe.featureType == 'Annotation'}

    @property
    def tables(self) -> dict[str, Table[Any]]:
        """A mapping of table names to `Table` objects in the dataset root"""
        return self._tables or {}

    @property
    def relationships(self) -> RelationshipManager:
        """A Manager object for interacting with RelationshipClasses"""
        return RelationshipManager(self)

    @property
    def domains(self) -> DomainManager:
        return DomainManager(self)

    @property
    def schema(self) -> SchemaWorkspace:
        return json.load(convert_schema(self, 'JSON'))

    def export_rules(self, rule_dir: Path|str) -> Iterator[AttributeRule]:
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
            yield from feature_class.attribute_rules.export_rules(Path(rule_dir))

    @overload
    def import_rules(self, rule_dir: Path | str, *, skip_fail: Literal[True]) -> Iterator[AttributeRule | Exception]: ...
    @overload
    def import_rules(self, rule_dir: Path | str, *, skip_fail: Literal[False]) -> Iterator[AttributeRule]: ...
    @overload
    def import_rules(self, rule_dir: Path | str, *, skip_fail: Literal[False]=False) -> Iterator[AttributeRule]: ...
    def import_rules(self, rule_dir: Path | str, 
                     *, 
                     skip_fail: bool=False) -> Iterator[AttributeRule | Exception]:
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
                    yield from feature_class.attribute_rules.import_rules(rule_dir / feature_class.name)
                except Exception as e:
                    if skip_fail:
                        print(f'Failed to import rules for {feature_class.name}: \n\t{e.__notes__}\n\t{e}')
                        yield e
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
        
        self._relationships = {}
        for root, _, rels in Walk(str(self.conn), datatype=['RelationshipClass']):
            root = Path(root)
            for rel in rels:
                # Backlink Datasets to parent
                if self.parent and rel in self.parent:
                    self._relationships[rel] = self.parent.relationships[rel]
                else:
                    self._relationships[rel] = Relationship(self, root / rel)
        
        # Handle datasets last to allow for backlinking     
        self._datasets = {}
        for root, ds, _ in Walk(str(self.conn), datatype=['FeatureDataset']):
            root = Path(root)
            self._datasets.update({d: Dataset(root / d, parent=self) for d in ds})
    
    def __getitem__(self, key: str) -> FeatureClass | Table[Any] | Dataset[Any] | Relationship:
        if ret := self.tables.get(key) or self.feature_classes.get(key) or self.datasets.get(key) or self.relationships.get(key):
            return ret
        raise KeyError(f'{key} is not a child of {self.conn.stem}')
        
    def get(self, key: str, default: _Default=None) -> FeatureClass | Table[Any] | Dataset[Any] | Relationship | _Default:
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
        
    def __iter__(self) -> Iterator[FeatureClass | Table[Any] | Dataset[Any] | Relationship]:
        for feature_class in self.feature_classes.values():
            yield feature_class
            
        for table in self.tables.values():
            yield table
            
        for dataset in self.datasets.values():
            yield from dataset
        
        for relationship in self.relationships:
            yield relationship
    
    def __len__(self) -> int:
        return sum(1 for _ in self)
           
    def __repr__(self) -> str:
        return (
            "Dataset("
            f"{self.name}, "
            "{"
            f"Features: {len(self.feature_classes)}, "
            f"Tables: {len(self.tables)}, "
            f"Datasets: {len(self.datasets)}, "
            f"Relationships: {len(self.relationships)}"
            "})"
        )
    
    def __str__(self) -> str:
        return self.__fspath__()
    
    def __fspath__(self) -> str:
        return str(self.conn.resolve())

    def export_schema_module(self, out_loc: Path|str, 
                           *,
                           tables: bool = True,
                           featureclasses: bool = True,
                           mod_doc: str | None = None,
                           fallback_type: type = object,
                           docs: dict[str, dict[str, str]] | None = None,
                           include_shape_token: bool = True,
                           include_oid_token: bool = True,
                           default_doc: Callable[[Field], str] | None | Literal['nodoc'] = None
        ) -> None:
        """Export the workspace to a python schema file that uses TypedDict and Annotated 
        to store field definitions. This is similar to Pydantic models, but these can be injested by
        Table and FeatureClass objects to type their iterators
        
        Args:
            tables: Include all table schemas in output
            featureclasses: Include all featureclasses in output 
            out_loc: The filepath of the output module (e.g. `<root>/schemas/db_schema.py`)
            mod_doc: Optional module documentation to include at the top of the file (default: `{self.name} Schema`)
            fallback_type: Default type for any fieldtype that can't be mapped to a Python type
            docs: Optional docs for each feature class in the format `{'Feature': {'Field': 'Field Doc', ...}, ...}`
            include_shape_token: Include @SHAPE in output schema (will inherit from FC shape)
            include_oid_token: Include the @OID token in the output schema
            default_doc: Optional default docstring func for fields (`'nodoc'` will exclude docstring from output)
        Note:
            If the supplied out_loc is not a valid `.py` python file, a python file with the name 
            `{self.name}_schema.py` will be generated there. Intermediate folders will be created if 
            they do not exist. 
        """
        from .schemas.fields import SCHEMA_IMPORTS
        if mod_doc:
            mod_doc = SCHEMA_IMPORTS.format(mod_doc)
        else:
            mod_doc = SCHEMA_IMPORTS.format(f'{self.name} Schema')
        
        out_loc = Path(out_loc)
        if out_loc.suffix != '.py':
            out_loc = out_loc / f'{self.name}.py'
        out_loc.parent.mkdir(exist_ok=True, parents=True)
        
        _items: list[FeatureClass | Table] = []
        if featureclasses:
            _items.extend(list(self.feature_classes.values()))
        if tables:
            _items.extend(list(self.tables.values()))
            
        with out_loc.open('wt') as fl:
            fl.write(mod_doc)
            for item in _items:
                # Extract any supplied FC docs
                doc = docs.get(item.name) if docs else None
                fl.write(
                    item.get_schema(
                        fallback_type=fallback_type, 
                        docs=doc, 
                        include_shape_token=include_shape_token, 
                        include_oid_token=include_oid_token,
                        default_doc=default_doc,
                    )
                )
                fl.write('\n\n')
        

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
            json.dump(schema, f, indent=2)
        return outfile
    
    @classmethod
    def from_schema(cls, schema: Path|str, out_loc: Path|str, gdb_name: str, 
                    *,
                    remove_rules: bool=False) -> Dataset[Any]:
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
    
    def __init__(self, dataset: Dataset[Any]) -> None:
        self._dataset = dataset
        
    @property
    def dataset(self) -> Dataset[Any]:
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
        usage = self.usage()
        return {
            name: domain
            for name, domain in self.domain_map.items() 
            if name not in usage
            or not usage[name]
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

def convert_cardinality(arg: str) -> Literal['ONE_TO_ONE', 'ONE_TO_MANY', 'MANY_TO_MANY']:
    if arg == 'OneToOne':
        return 'ONE_TO_ONE'
    if arg == 'OneToMany':
        return 'ONE_TO_MANY'
    if arg == 'ManyToMany':
        return 'MANY_TO_MANY'
    else:
        raise ValueError(f'{arg} is not a valid relationsip type')
  
class Relationship:
    def __init__(self, parent: Dataset[Any], path: Path|str) -> None:
        self.parent = parent
        self.path = path
    
    @cached_property
    def describe(self) -> RelationshipClass:
        return Describe(str(self.path)) # pyright: ignore[reportUnknownVariableType]

    @property
    def name(self) -> str:
        return self.describe.name

    @property
    def cardinality(self) -> Literal['ONE_TO_ONE', 'ONE_TO_MANY', 'MANY_TO_MANY']:
        return convert_cardinality(self.describe.cardinality)

    @property
    def settings(self) -> RelationshipOpts:
        return RelationshipOpts(
            origin_table=self.describe.originClassNames[0],
            destination_table=self.describe.destinationClassNames[0],
            out_relationship_class=self.name,
            relationship_type='COMPOSITE' if self.describe.isComposite else 'SIMPLE',
            forward_label=self.describe.forwardPathLabel,
            backward_label=self.describe.backwardPathLabel,
            message_direction=self.describe.notification.upper(), # type: ignore
            cardinality=convert_cardinality(self.describe.cardinality),
            attributed= 'ATTRIBUTED' if self.describe.isAttributed else 'NONE',
            origin_primary_key=self.origin_keys['OriginPrimary'],
            origin_foreign_key=self.origin_keys['OriginForeign'],
            destination_primary_key=self.destination_keys['DestinationPrimary'],
            destination_foreign_key=self.destination_keys['DestinationForeign'],
        )

    @property
    def origins(self) -> list[FeatureClass | Table[Any]]:
        """Origin FeatureClass/Table objects"""
        return [
            self.parent.feature_classes.get(origin) or self.parent.tables[origin]
            for origin in self.describe.originClassNames 
            if origin in self.parent
        ]
     
    @property
    def origin_keys(self) -> dict[Literal['OriginPrimary', 'OriginForeign'], str]:
        """Mapping of origin Primary and Foreign keys"""
        keys = {
            'OriginPrimary': '',
            'OriginForeign': '',
        }
        for field, key_type, _ in self.describe.originClassKeys:
            if key_type == 'OriginForeign':
                keys['OriginForeign'] = field
            elif key_type == 'OriginPrimary':
                keys['OriginPrimary'] = field
        return keys # pyright: ignore[reportReturnType]
        
    @property
    def destinations(self) -> list[FeatureClass | Table[Any]]:
        """Destination FeatureClass/Table objects"""
        return [
            self.parent.feature_classes.get(dest) or self.parent.tables[dest]
            for dest in self.describe.destinationClassNames 
            if dest in self.parent
        ]

    @property
    def destination_keys(self) -> dict[Literal['DestinationPrimary', 'DestinationForeign'], str]:
        """Mapping of destination Primary and Foreign keys"""
        keys = {
            'DestinationPrimary': '',
            'DestinationForeign': '',
        }
        for field, key_type, _ in self.describe.destinationClassKeys:
            if key_type == 'DestinationForeign':
                keys['DestinationForeign'] = field
            elif key_type == 'DestinationPrimary':
                keys['DestinationPrimary'] = field
        return keys # pyright: ignore[reportReturnType]
    
    def add_rule(self, **options: Unpack[RelationshipAddRuleOpts]) -> None:
        options['in_rel_class'] = str(self.path)
        AddRuleToRelationshipClass(**options)
    
    def remove_rule(self, **options: Unpack[RelationshipRemoveRuleOpts]) -> None:
        options['in_rel_class'] = str(self.path)
        RemoveRuleFromRelationshipClass(**options)
        
    def delete(self) -> None:
        """Delete the relationship"""
        Delete(str(self.path), 'RelationshipClass')
    
    def update(self, **options: Unpack[RelationshipOpts]) -> None:
        """Update the relationship class"""
        rel_opts = self.settings
        self.delete()
        rel_opts.update(options)
        CreateRelationshipClass(**rel_opts)

class RelationshipManager:
    def __init__(self, parent: Dataset[Any]) -> None:
        self.parent = parent
    
    @property
    def relationships(self) -> dict[str, Relationship]:
        return self.parent._relationships # pyright: ignore[reportPrivateUsage]
    
    @property
    def names(self) -> list[str]:
        return list(self.relationships.keys())
    
    def create(self, **options: Unpack[RelationshipOpts]) -> None:
        """Create a relationship"""
        CreateRelationshipClass(**options)
    
    def delete(self, name: str) -> RelationshipOpts | None:
        """Delete the relationship and return the settings so it can be made again"""
        rel = self.get(name)
        if rel is None:
            return None
        settings = rel.settings
        rel.delete()
        return settings
    
    def __len__(self) -> int:
        return len(self.relationships)
    
    def __iter__(self) -> Iterator[Relationship]:
        for rel in self.relationships.values():
            yield rel
    
    def __getitem__(self, key: str) -> Relationship:
        if key in self.relationships:
            return self.relationships[key]
        raise KeyError(f'{key} not found in {self.parent.name} Relationships')
    
    def get(self, key: str, default: _Default=None) -> Relationship | _Default:
        try:
            return self[key]
        except KeyError:
            return default