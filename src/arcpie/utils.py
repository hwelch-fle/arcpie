"""Module for internal utility functions to share between modules"""
from __future__ import annotations
from pathlib import Path
import json

from typing import (
    Any,
)

from .featureclass import (
    Table,
    FeatureClass,
    where,
    count,
)

from .database import (
    Dataset,
)

from .project import (
    Map,
    Project,
    Layer,
    Table as StandaloneTable,
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
             _digit_chars = []
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
        