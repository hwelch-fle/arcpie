"""Module for internal utility functions to share between modules"""
from __future__ import annotations
import builtins
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
import json

from tempfile import TemporaryDirectory
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from arcpy import (
    AddMessage,
    AddWarning,
    AddError,
    PointGeometry,
    Polyline,
)

from .featureclass import (
    Table,
    FeatureClass,
    where,
    count,
)

if TYPE_CHECKING:
    from .database import (
        Dataset,
    )

from .project import (
    Map,
    Project,
    Layer,
    Table as StandaloneTable,
)

from arcpy.management import (
    ConvertSchemaReport, # pyright: ignore[reportUnknownVariableType]
    GenerateSchemaReport, # pyright: ignore[reportUnknownVariableType]
)

def nat(val: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Natural sort key for use in string sorting
    
    Args:
        val (str): A value that you want the natural sort key for
    
    Returns:
        (tuple[tuple[int, ...], tuple[str, ...]): A tuple containing all numeric and 
        string components in order of appearance. Best used as a sort key
    
    Usage:
        ```python
        >>> pages = ['P-1.3', 'P-2.11', ...]
        >>> pages.sort(key=nat)
        >>> print(pages)
        ['P-1.1', 'P-1.2', ...]
        ```
    """
    _digits: list[int] = []
    _alpha: list[str] = []
    _digit_chars: list[str] = []
    for s in val:
       if s.isdigit():
          _digit_chars.append(s)
       else:
          _alpha.append(s)
          if _digit_chars:
             _digits.append(int(''.join(_digit_chars)))
             _digit_chars.clear()
    if _digit_chars:
       _digits.append(int(''.join(_digit_chars)))
    return tuple(_digits), tuple(_alpha)

def get_subtype_count(fc: Table | FeatureClass[Any], drop_empty: bool=False) -> dict[str, int]:
    """Get the subtype counts for a Table or FeatureClass
    
    Args:
        fc (Table | FeatureClass): The Table/FeatureClass you want subtype counts for
        drop_empty (bool): Drop any counts that have no features from the output dictionary (default: False)
    
    Returns:
        (dict[str, int]): A mapping of subtype name to subtype count
    """
    return {
        subtype['Name']: cnt
        for code, subtype in fc.subtypes.items() 
        if fc.subtype_field # has Subtypes
        and (
            (cnt := count(fc[where(f'{fc.subtype_field} = {code}')])) # Get count
            or drop_empty # Drop Empty counts?
        )
    }

def print(*values: object,
          sep: str = " ",
          end: str = "\n",
          file: Any = None,
          flush: bool = False,
          severity: Literal['INFO', 'WARNING', 'ERROR']|None = None) -> None:
    """ Print a message to the ArcGIS Pro message queue and stdout
    set severity to 'WARNING' or 'ERROR' to print to the ArcGIS Pro message queue with the appropriate severity
    """

    # Print the message to stdout
    builtins.print(*values, sep=sep, end=end, file=file, flush=flush)
    
    end = "" if end == '\n' else end
    message = f"{sep.join(map(str, values))}{end}"
    # Print the message to the ArcGIS Pro message queue with the appropriate severity
    match severity:
        case "WARNING":
            AddWarning(f"{message}")
        case "ERROR":
            AddError(f"{message}")
        case _:
            AddMessage(f"{message}")
    return

def get_subtype_counts(gdb: Dataset, *, drop_empty: bool=False) -> dict[str, dict[str, int]]:
    """Get a mapping of subtype counts for all featureclasses that have subtypes in the provided Dataset
    
    Args:
        gdb (Dataset): The Dataset instance to get subtype counts for
        drop_empty (bool): Drop any counts that have no features from the output dictionary (default: False)
    
    Returns:
        (dict[str, dict[str, int]]): A mapping of FeatureClass -> SubtypeName -> SubtypeCount
    
    Usage:
        ```python
        >>> get_subtype_counts(Dataset('<path/to/gdb>', drop_empty=True))
        {
            'FC1': 
                {
                    'Default': 10
                    'Subtype 1': 6
                    ...
                },
            ...
        }
        ```  
    """
    feats: list[Table] = [*gdb.feature_classes.values(), *gdb.tables.values()]
    return {
        fc.name: counts
        for fc in feats
        if (counts := get_subtype_count(fc))
        or not drop_empty
    }
    
def export_project_lyrx(project: Project, out_dir: Path, *, indent: int=4, sort: bool=False, skip_empty: bool=True) -> None:
    """Pull all layers from a project file and output them in a directory as lyrx files
    
    Args:
        project (Project): The `arcpie.Project` instance to export
        out_dir (Path|str): The target directory for the layer files
        indent (int): Indentation level of the ouput files (default: 4)
        sort (bool): Sort the output file by key name (default: False)
        skip_empty (bool): Skips writing empty lyrx files for layers with no lyrx data (default: True)
    
    Usage:
        ```python
        >>> export_project_lyrx(arcpie.Project('<path/to/aprx>'), '<path/to/output_dir>')
        ```
    
    Note:
        Output structure will match the structure of the project:
        `Map -> Group -> Layer`
        Where each level is a directory. Group Layers will have a directory entry with individual
        files for each layer they contain, as well as a single layerfile that contains all their 
        child layers.
    """
    out_dir = Path(out_dir)
    for map in project.maps:
        map_dir = out_dir / map.unique_name
        for layer in map.layers:
            _lyrx = getattr(layer, 'lyrx', None)
            if _lyrx is None:
                print(f'{(layer.cim_dict or {}).get("type")} is invalid!')
                continue
            if skip_empty and not _lyrx:
                continue
            out_file = (map_dir / layer.longName).with_suffix('.lyrx')
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(_lyrx, indent=indent, sort_keys=sort), encoding='utf-8')

def export_project_maps(project: Project, out_dir: Path|str, *, indent: int=4, sort: bool=False) -> None:
    """Pull all layers from a project file and output them in a directory as mapx files
    
    Args:
        project (Project): The `arcpie.Project` instance to export
        out_dir (Path|str): The target directory for the mapx files
        indent (int): Indentation level of the ouput files (default: 4)
        sort (bool): Sort the output file by key name (default: False)
    
    Usage:
        ```python
        >>> export_project_maps(arcpie.Project('<path/to/aprx>'), '<path/to/output_dir>')
        ```
    """
    out_dir = Path(out_dir)
    for map in project.maps:
        map_dir = out_dir / rf'{map.unique_name}'
        out_file = map_dir.with_suffix(f'{map_dir.suffix}.mapx') # handle '.' in map name
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(map.mapx, indent=indent, sort_keys=sort), encoding='utf-8')
        
def build_mapx(source_map: Map, layers: list[Layer], tables: list[StandaloneTable]) -> dict[str, Any]:
    _base_map = source_map.mapx
    
    # Remove existing definitions
    _base_map.pop('layerDefinition', None)
    _base_map.pop('tableDefinitions', None)
    
    # Remove existing CIM paths
    _map_def: dict[str, Any] = _base_map['mapDefinition']
    _map_def.pop('layers', None)
    _map_def.pop('standaloneTables', None)
    
    if layers:
        _map_def['layers'] = [l.URI for l in layers]
        _base_map['layerDefinitions'] = [l.cim_dict for l in layers]
    
    if tables:
        _map_def['standaloneTables'] = [t.URI for t in tables]
        _base_map['tableDefinitions'] = [t.cim_dict for t in tables]
    
    return _base_map

from .schemas import SchemaWorkspace, SchemaDataset

def convert_schema(schema: Dataset[Any]|Path|str, to: Literal['JSON', 'XLSX', 'HTML', 'PDF', 'XML']='JSON') -> BytesIO:
    """Convert a Schema from one format to another
    
    Args:
        schema (Dataset|Path|str): Path to the schemafile or Dataset to convert
        to (Literal['JSON', 'XLSX', 'HTML', 'PDF', 'XML']): Target format (default: 'JSON')
        
    Yields:
        bytes: Raw bytes object containing the schema file
    """
    with TemporaryDirectory(suffix=to) as temp:
        temp = Path(temp)
        if not isinstance(schema, (Path, str)):
            # Convert Dataset to report
            schema, = GenerateSchemaReport(str(schema.conn), str(temp), 'json_schema', 'JSON')
        schema = Path(schema)
        conversion, = ConvertSchemaReport(str(schema), str(temp), 'out', to)
        return BytesIO(Path(conversion).read_bytes())

def patch_schema_rules(schema: SchemaWorkspace|Path|str, 
                       *,
                       remove_rules: bool=False) -> SchemaWorkspace:
    """Patch an exported Schema doc by re-linking attribute rules to table names
    
    Args:
        schema (Path|str): The input schema to patch
        remove_rules (bool): Remove attribute rules from the schema (default: False)
    
    Returns:
        SchemaWorkspace: A patched schema dictionary
    """
    # Load schema
    if isinstance(schema, dict):
        workspace = schema
    else:
        schema = Path(schema)
        if not schema.suffix == '.json':
            raise ValueError(f'Schema Patching can only be done on json schemas!, got {schema.suffix}')
        workspace: SchemaWorkspace = json.load(schema.open(encoding='utf-8'))
        
    # Get all root Features
    features = [
        ds 
        for ds in workspace['datasets'] 
        if 'datasets' not in ds
    ]
    # Get all features that live in a FeatureDataset
    # Use listcomp since typing is weird
    [
        features.extend(ds['datasets']) 
        for ds in workspace['datasets'] 
        if 'datasets' in ds
    ]
    # Get a translation dictionary of catalogID -> name
    guid_to_name = {
        fc['catalogID']: fc['name'] 
        for fc in features 
        if 'catalogID' in fc
    }
    for feature in filter(lambda fc: 'attributeRules' in fc, features):
        feature: SchemaDataset
        if remove_rules:
            feature['attributeRules'] = []
            continue
        rules = feature['attributeRules']
        for rule in rules:
            script = rule['scriptExpression']
            for guid, name in guid_to_name.items():
                script = script.replace(guid, name)
            rule['scriptExpression'] = script
    return workspace

def split_at_points(lines: FeatureClass[Polyline, Any], points: FeatureClass[PointGeometry, Any], 
                *, 
                buffer: float=0.0,
                min_len: float=0.0) -> Iterator[tuple[int, Polyline]]:
    """Split lines at provided points
    
    Args:
        lines (FeatureClass[Polyline]): Line features to split
        points (FeatureClass[PointGeometry]): Points to split on
        buffer (float): Split buffer in feature units (default: 0.0 [exact])
        min_len (float): Minumum length for a new line in feature units (default: 0.0)
    
    Yields:
        ( tuple[int, Polyline]] ): Tuples of parent OID and child shape
    
    Warning:
        When splitting features in differing projections, the point features will be projected
        into the spatial reference of the line features.
    
    Example:
        ```python
        >>> # Simple process for splitting lines in place
        ... 
        >>> # Initialize a set to capture the removed ids
        >>> removed: set[int] = set()
        >>> with lines.editor:
        ...     # Insert new lines
        ...     with lines.insert_cursor('SHAPE@') as cur:
        ...         for parent, new_line in split_at_points(lines, points):
        ...             cur.insertRow([new_line])
        ...             removed.add(parent) # Add parent ID to removed
        ...     # Remove old lines (if you're inserting to the same featureclass)
        ...     with lines.update_cursor('OID@') as cur:
        ...         for _ in filter(lambda r: r[0] in removed, cur):
        ...             cur.deleteRow() 
        ```
    """
    line_iter: Iterator[tuple[Polyline, int]] = lines[('SHAPE@', 'OID@')]
    for line, oid in line_iter:
        int_points: list[PointGeometry] = []
        with points.reference_as(line.spatialReference), points.fields_as('SHAPE@'):
            int_points = [r['SHAPE@'] for r in points[line.buffer(buffer)]]
        
        if len(int_points) == 0 or all(p.touches(line) for p in int_points):
            continue
        
        prev_measure = 0.0
        measures = sorted(line.measureOnLine(p) for p in int_points) + [line.length]
        for measure in measures:
            seg = line.segmentAlongLine(prev_measure, measure)
            prev_measure = measure
            if seg and seg.length >= (min_len or 0):
                yield oid, seg