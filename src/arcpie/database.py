from __future__ import annotations

from functools import cached_property
import json
from pathlib import Path

from tempfile import TemporaryDirectory

from collections.abc import (
    Callable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Unpack,
    overload,
)

from arcpy import Describe, SpatialReference # type: ignore (incorrect hinting from arcpy)

from arcpie.cursor import Field

from ._types import (
    AttributeRule,
    RelationshipOpts,
    RelationshipRemoveRuleOpts,
    RelationshipAddRuleOpts,
)
from arcpie.schema.workspace import SchemaWorkspace
from arcpie.schema.field import GeoType
from arcpie.schema.domain import DomainManager

from .featureclass import (
    Table,
    FeatureClass,
)

from .utils import patch_schema_rules, convert_schema

from arcpy.da import (
    Walk,
    Editor,
)

from arcpy.management import (
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

# Type Alias to allow setting Spatial Reference using WKID int
type WKID = int

WGS84 = SpatialReference(4326)

SYSTEM_FIELDS = ['objectid', 'shape', 'shape_area', 'shape_length']

class Dataset[_S = Mapping[str, Any]]:
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
        if parent is None and self.conn.suffix != '.gdb':
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
    def feature_classes(self) -> dict[str, FeatureClass[Any, Any]]:
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

    @cached_property
    def schema(self) -> SchemaWorkspace:
        return json.load(convert_schema(self, 'JSON'))

    @property
    def editor(self) -> Editor:
        return Editor(str(self.conn))

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
        
        # Clear the schema cache
        if hasattr(self, 'schema'):
            del self.schema
    
    def __getitem__(self, key: str) -> FeatureClass[Any, Any] | Table[Any] | Dataset[Any] | Relationship:
        if ret := self.tables.get(key) or self.feature_classes.get(key) or self.datasets.get(key) or self.relationships.get(key):
            return ret
        raise KeyError(f'{key} is not a child of {self.conn.stem}')
        
    def get[D](self, key: str, default: D=None) -> FeatureClass[Any, Any] | Table[Any] | Dataset[Any] | Relationship | D:
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
        
    def __iter__(self) -> Iterator[FeatureClass[Any, Any] | Table[Any] | Dataset[Any] | Relationship]:
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
            f"Relationships: {len(self.relationships)}, "
            f"Domains: {len(self.domains)}"
            "})"
        )
    
    def __str__(self) -> str:
        return self.__fspath__()
    
    def __fspath__(self) -> str:
        return str(self.conn.resolve())

    def create_featureclass(self, name: str, geometry_type: GeoType | None, feature_dataset: str | None = None,
                            *,
                            template: FeatureClass | None = None,
                            has_m: bool | None = None,
                            has_z: bool | None = None,
                            spatial_reference: SpatialReference | WKID = WGS84,
                            config_keyword: str | None = None,
                            alias: str | None = None,
                            oid_type: Literal['64_BIT', '32_BIT'] | None = '64_BIT',
                            _ensure_dataset: bool = True,
        ) -> FeatureClass:
        """Create a new FeatureClass in the Dataset"""
        if feature_dataset:
            path = str(self.conn / feature_dataset)
            # Since we're modifying the dataset, we need to re-index
            if _ensure_dataset:
                self.walk()
                if feature_dataset not in self.datasets:
                    self.create_feature_dataset(feature_dataset, spatial_reference=spatial_reference)
        else:
            path = str(self.conn)
        return FeatureClass(
            CreateFeatureclass(
                out_path=path,
                out_name=name,
                geometry_type=geometry_type,
                template=str(template.path) if template else None,
                has_m='SAME_AS_TEMPLATE' if has_m is None and template is not None else ('ENABLED' if has_m else 'DISABLED'),
                has_z='SAME_AS_TEMPLATE' if has_z is None and template is not None else ('ENABLED' if has_z else 'DISABLED'),
                spatial_reference=spatial_reference,
                config_keyword=config_keyword,
                out_alias=alias,
                oid_type='SAME_AS_TEMPLATE' if oid_type is None and template is not None else oid_type,
            )[0]
        )
    
    def create_table(self, name: str,
                     *,
                     template: Table | None = None,
                     config_keyword: str | None = None,
                     alias: str | None = None,
                     oid_type: Literal["64_BIT", "32_BIT"] | None = '64_BIT',
        ) -> Table:
        """Create a new Table in the Dataset
        
        Args:
            name: The name of the new table
            template: A Table object to use as a template
            config_keyword: A keyword to pass to the database engine for table setup
            alias: An alias for the table
            oid_type: The size of OID to use. If template is set and this is `None`, template OID type is used
        """
        
        return Table(
            CreateTable(
                out_path=str(self.conn),
                out_name=name,
                template=str(template.path) if template else None,
                config_keyword=config_keyword,
                out_alias=alias,
                oid_type='SAME_AS_TEMPLATE' if template and oid_type is None else oid_type
            )[0]
        )

    def create_feature_dataset(self, name: str, *, spatial_reference: SpatialReference | WKID = WGS84) -> Dataset:
        """Create a FeatureDataset (cannot be done if the parent Dataset is already a FeatureDataset!)
        
        Args:
            name: The name for the new feature dataset (must be unique in parent dataset!)
            spatial_reference: An optional spatial reference to use for the dataset (default `WGS84`/`EPSG:4326`)
        
        Returns:
            A new Dataset object parented to this Dataset
        """
        if isinstance(spatial_reference, int):
            spatial_reference = SpatialReference(spatial_reference)
        return Dataset(CreateFeatureDataset(str(self.conn), name, spatial_reference=spatial_reference)[0], parent=self)

    # TODO: This whole flow is really tightly coupled. I'd like to find a better way
    def export_schema_module(self, out_loc: Path|str, 
                           *,
                           tables: bool | Sequence[str] = True,
                           featureclasses: bool | Sequence[str] = True,
                           datasets: bool | Sequence[str] = True,
                           mod_doc: str | None = None,
                           fallback_type: type = object,
                           docs: dict[str, dict[str, str]] | None = None,
                           include_shape_token: bool = True,
                           include_oid_token: bool = True,
                           default_doc: Callable[[Field], str] | None | Literal['nodoc'] = None,
                           skip_annotations: bool = False,
        ) -> None:
        """Export the workspace to a python schema file that uses TypedDict and Annotated 
        to store field definitions. This is similar to Pydantic models, but these can be ingested by
        Table and FeatureClass objects to type their iterators
        
        Args:
            tables: Include table schemas in output (Only specified names if a sequence is provided)
            featureclasses: Include featureclasses in output (Only specified names if a sequence is provided)
            datasets: Include schemas for datasets in the output (Only specified names if a sequence is provided)
            out_loc: The filepath of the output module (e.g. `<root>/schemas/db_schema.py`)
            mod_doc: Optional module documentation to include at the top of the file (default: `{self.name} Schema`)
            fallback_type: Default type for any fieldtype that can't be mapped to a Python type
            docs: Optional docs for each feature class in the format `{'Feature': {'Field': 'Field Doc', ...}, ...}`
            include_shape_token: Include @SHAPE in output schema (will inherit from FC shape)
            include_oid_token: Include the @OID token in the output schema
            default_doc: Optional default docstring func for fields (`'nodoc'` will exclude docstring from output)
            skip_annotations: Don't export schema for Annotation Features
        Note:
            If the supplied out_loc is not a valid `.py` python file, a python file with the name 
            `{self.name}_schema.py` will be generated there. Intermediate folders will be created if 
            they do not exist. 
        """
        from .schema.field import SCHEMA_IMPORTS
        if mod_doc:
            mod_doc = SCHEMA_IMPORTS.format(mod_doc)
        else:
            mod_doc = SCHEMA_IMPORTS.format(f'{self.name} Schema')
        
        out_loc = Path(out_loc)
        if out_loc.suffix != '.py':
            out_loc = out_loc / f'{self.name}.py'
        out_loc.parent.mkdir(exist_ok=True, parents=True)
        
        _items: list[FeatureClass | Table] = []
        
        # Gather all requested FeatureClasses and Tables
        _features = []
        if featureclasses:
            # Skip annotations since they have additional interfaces that aren't modeled
            if isinstance(featureclasses, Sequence):
                _features = [
                    fc 
                    for fc in self.feature_classes.values()
                    if fc.name in featureclasses
                ]
            else:
                _features = list(self.feature_classes.values())
            _items.extend(_features)
        
        _tables = []
        if tables:
            if isinstance(tables, Sequence):
                _tables = [
                    tbl
                    for tbl in self.tables.values()
                    if tbl.name in tables
                ]
            else:
                _tables = list(self.tables.values())
            _items.extend(_tables)
        
        _datasets = []
        if datasets:
            if isinstance(datasets, Sequence):
                _datasets = [
                    ds 
                    for ds in self.datasets.values() 
                    if ds.name in datasets
                ]
            else:
                _datasets = list(self.datasets.values())
        with out_loc.open('wt') as fl:
            fl.write(mod_doc)
            
            # Notate root for later parsing operations
            fl.write("# Entry Point for parser\n")
            fl.write(f'SCHEMA_ROOT = "{self.name}"\n\n')
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
            
            if _datasets:
                fl.write('# Dataset Definitions\n\n')
            
            ds_items: set[str] = set()
            for ds in _datasets:
                _ds_children = list(filter(lambda i: i.name in ds, _items))
                fl.write(f"class {ds.name}(TypedDict):\n")
                fl.write('    """FeatureDataset"""\n\n')
                for item in _ds_children:
                    fl.write(f"    {item.name}: {item.name}\n")
                    ds_items.add(item.name)
                fl.write('\n\n')
            
            fl.write("# Root Schema\n\n")
            
            fl.write(f"class {self.name}(TypedDict):\n")
            fl.write('    """Dataset"""\n\n')
            for item in filter(lambda i: i.name not in ds_items, _items):
                fl.write(f"    {item.name}: {item.name}\n")
            for ds in _datasets:
                fl.write(f"    {ds.name}: {ds.name}\n")
            fl.write('\n')
    
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
    
    @classmethod
    def from_schema_module(cls, out_loc: Path | str, schema_module: ModuleType, 
                           *,
                           spatial_reference: SpatialReference | WKID = WGS84,
                           overwrite: bool = False,
                           ) -> Generator[FeatureClass | Table, None, Dataset[Any]]:
        """Build a new GDB from an existing schema module generated with `export_schema_module`
        
        Args:
            out_loc: Destination for the generated GDB
            schema_module: The module containing all table definitions
            spatial_reference: The Spatial Reference to generate the database in (default: `WGS84`/`EPSG:4326`)
            overwrite: If the target database exists, overwrite it
        
        Yields:
            FeatureClasses/Tables as they are created for monitoring purposes
        
        Returns:
            A new Dataset object built from the schema
        
        Raises:
            FileExistsError: When the target directory exists and `overwrite` is set to `False`
        
        Example:
            >>> import my_database_schema
            >>> new_ds = Dataset.from_schema_module('new_database.gdb', my_database_schema, 3857)
        """
        # Defer imports
        from .schema.field import parse_hierarchy
        from typing import is_typeddict
        schema_root = getattr(schema_module, 'SCHEMA_ROOT', None)
        if not schema_root:
            raise ValueError(
                f'A SCHEMA_ROOT global must be declared in the schema module '
                '(this is the name of last item generated by the export)'
            )
        
        root_dict: type | None = getattr(schema_module, schema_root, None)
        if not root_dict or not is_typeddict(root_dict):
            raise ValueError('SCHEMA_ROOT must be a TypedDict')
        if not (root_dict.__doc__ or '').startswith('Dataset'):
            raise ValueError('SCHEMA_ROOT must have a docstring with `Dataset` as the first line')
        
        
        out_loc = Path(out_loc)
        if out_loc.suffix != '.gdb':
            # Enforce '.gdb' suffix, and allow other suffixes:
            # e.g. ../my_database.new -> ../my_database.new.gdb
            out_loc = out_loc.with_suffix(out_loc.suffix + '.gdb')
        if out_loc.exists():
            if not overwrite:
                raise FileExistsError(
                    f'{out_loc} Exists! '
                    'To overwrite it, set the `overwrite` flag to True'
                )
            else:
                # Import rmtree to simplify gdb directory removal
                from shutil import rmtree
                rmtree(out_loc)
        
        # Create and bind the GDB
        CreateFileGDB(str(out_loc.parent), out_loc.name, 'CURRENT')
        ds = cls(out_loc)
        
        hierarchy = parse_hierarchy(root_dict, skip_annos=True)
        for child_name, child_def in hierarchy.items():
            child_def: tuple[GeoType | None, dict[str, Field]] | dict[str, Any]
            
            # Build FeatureClasses/Tables
            if isinstance(child_def, tuple):
                shape_type, fields = child_def
                if shape_type is None:
                    table = ds.create_table(child_name)
                    for field_name, field_props in fields.items():
                        if field_name.lower() in SYSTEM_FIELDS:
                            continue
                        if field_name == 'GlobalID':
                            table.add_globalids()
                            continue
                        try:
                            table.add_field(field_name, **field_props)
                        except Exception as e:
                            print(f'{field_name}: ', e)
                    yield table
                else:
                    fc = ds.create_featureclass(child_name, geometry_type=shape_type, spatial_reference=spatial_reference)
                    for field_name, field_props in fields.items():
                        if field_name.lower() in SYSTEM_FIELDS:
                            continue
                        if field_name == 'GlobalID':
                            fc.add_globalids()
                            continue
                        try:
                            fc.add_field(field_name, **field_props)
                        except Exception as e:
                            print(f'{field_name}: ', e)
                    yield fc
            
            # Parse FeatureDataset
            if isinstance(child_def, dict):
                ds_name = child_name
                ds.create_feature_dataset(ds_name, spatial_reference=spatial_reference)
                for fc_name, fc_def in child_def.items():
                    fc_def: tuple[GeoType | None, dict[str, Field]]
                    shape_type, fields = fc_def
                    fc = ds.create_featureclass(fc_name, shape_type, ds_name, _ensure_dataset=False)
                    for field_name, field_props in fields.items():
                        if field_name.lower() in SYSTEM_FIELDS:
                            continue
                        if field_name == 'GlobalID':
                            fc.add_globalids()
                            continue
                        try:
                            fc.add_field(field_name, **field_props)
                        except Exception as e:
                            print(f'{field_name}: ', e)
                    yield fc
        return ds
        
        
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
        return Describe(str(self.path)) # type: ignore (incorrect hinting from arcpy)

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
    
    def get[D](self, key: str, default: D=None) -> Relationship | D:
        try:
            return self[key]
        except KeyError:
            return default