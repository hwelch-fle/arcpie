from __future__ import annotations

"""Module for diffing ArcGIS objects"""

from typing import Literal
from arcpie.database import Dataset
from arcpie.project import Project

Diff = dict[Literal['added', 'removed', 'updated'], list[str]]

def feature_diff(source: Dataset, target: Dataset) -> Diff:
    """Get features that are added/removed from source/target
    
    Args:
        source (Dataset): The starting point of the delta
        target (Dataset): The ending point of the delta
    
    Returns:
        (Diff): FeatureClass delta
    """
    _diff: Diff = {}
    _diff['added'] = list(set(target.feature_classes) - set(source.feature_classes))
    _diff['removed'] = list(set(source.feature_classes) - set(target.feature_classes))
    _diff['updated'] = [
        fc for fc in source.feature_classes
        if fc in target
        and (
            set(target.feature_classes[fc].fields) != set(source.feature_classes[fc].fields)
            or not target.feature_classes[fc].py_types.values() == source.feature_classes[fc].py_types.values()
        ) 
    ]
    return _diff

def field_diff(source: Dataset, target: Dataset) -> dict[str, Diff]:
    """Get fields that are added/removed from source/target
    
    Args:
        source (Dataset): The starting point of the delta
        target (Dataset): The ending point of the delta
    
    Returns:
        (dict[str, Diff]): A Mapping of feature names to field deltas
    """
    _diffs: dict[str, Diff] = {}
    for fc_name, source_fc in source.feature_classes.items():
        if (target_fc := target.feature_classes.get(fc_name, None)) is not None:
            _diffs.setdefault(fc_name, {})
            _diffs[fc_name]['added'] = [f for f in target_fc.fields if f not in source_fc.fields]
            _diffs[fc_name]['removed'] = [f for f in source_fc.fields if f not in target_fc.fields]
            
            # Compare fields from matching features
            _to_compare = ('baseName', 'aliasName', 'defaultValue', 'domain', 
                           'editable', 'isNullable', 'length', 'name', 
                           'precision', 'required', 'scale', 'type')
            _source_fields = {f.baseName: f for f in source_fc.describe.fields}
            _target_fields = {f.baseName: f for f in target_fc.describe.fields}
            _diffs[fc_name]['updated'] = [
                f for f in _source_fields
                if f in _target_fields
                and not all(
                    getattr(_source_fields[f], attr) == getattr(_target_fields[f], attr)
                    for attr in _to_compare
                )
            ]
    return {k:v for k, v in _diffs.items() if v['added'] or v['removed'] or v['updated']}

def layer_diff(source: Project, target: Project) -> dict[str, Diff]:
    """Get layers that are added/removed from source/target
    
    Args:
        source (Project): The starting point of the delta
        target (Project): The ending point of the delta
    
    Returns:
        (dict[str, Diff]): A Mapping of map names to layer deltas
    """
    _diffs: dict[str, Diff] = {}
    for source_map in source.maps:
        map_name = source_map.name
        if (target_map := target.maps.get(source_map.name, None)) is not None:
            _diffs.setdefault(map_name, {})
            _diffs[map_name]['added'] = list(set(target_map.layers.names) - set(source_map.layers.names))
            _diffs[map_name]['removed'] = list(set(source_map.layers.names) - set(target_map.layers.names))
            _diffs[map_name]['updated'] = [
                l.name for l in source_map.layers
                if hasattr(l, 'name') 
                and l.name in target_map.layers.names
                and (source_cim := l.cim_dict)
                and (target_cim := target_map.layers[l.name].cim_dict)
                and any(
                    str(source_cim.get(k, 'Source')) != str(target_cim.get(k, 'Target'))
                    for k in ['renderer', 'labelClasses'] # Only compare symbology and labels
                )
            ]
    return {k:v for k, v in _diffs.items() if v['added'] or v['removed'] or v['updated']}

def table_diff(source: Project, target: Project) -> dict[str, Diff]:
    """Get tables that are added/removed/changed from source/target
    
    Args:
        source (Project): The starting point of the delta
        target (Project): The ending point of the delta
    
    Returns:
        (dict[str, Diff]): A Mapping of map names to table deltas
    """
    _diffs: dict[str, Diff] = {}
    for source_map in source.maps:
        map_name = source_map.name
        if (target_map := target.maps.get(source_map.name, None)) is not None:
            _diffs.setdefault(map_name, {})
            _diffs[map_name]['added'] = list(set(target_map.tables.names) - set(source_map.tables.names))
            _diffs[map_name]['removed'] = list(set(source_map.tables.names) - set(target_map.tables.names))
            _diffs[map_name]['updated'] = [
                t.name for t in source_map.tables
                if hasattr(t, 'name') 
                and t.name in target_map.tables.names
                and (source_cim := t.cim_dict)
                and (target_cim := target_map.tables[t.name].cim_dict)
                and any(
                    str(source_cim.get(k, 'Source')) != str(target_cim.get(k, 'Target'))
                    for k in ['displayField'] # Only compare display field
                )
            ]
    return {k:v for k, v in _diffs.items() if v['added'] or v['removed'] or v['updated']}

def attribute_rule_diff(source: Dataset, target: Dataset) -> dict[str, Diff]:
    """Get a diff of rules for matching FeatureClasses in the dataset
    
    Args:
        source (Dataset): The starting point of the delta
        target (Dataset): The ending point of the delta
        
    Returns:
        (dict[str, diff]): A Mapping of Features to rules with a delta type
    """
    
    _diffs: dict[str, Diff] = {}
    for fc_name, source_fc in source.feature_classes.items():
        if (target_fc := target.feature_classes.get(fc_name, None)) is not None:
            _diffs.setdefault(fc_name, {})
            _diffs[fc_name]['added'] = [rule for rule in target_fc.attribute_rules if rule not in source_fc.attribute_rules]
            _diffs[fc_name]['removed'] = [rule for rule in source_fc.attribute_rules if rule not in target_fc.attribute_rules]
            _diffs[fc_name]['updated'] = [
                rule_name
                for rule_name, rule in source_fc.attribute_rules.items() 
                if (t_rule := target_fc.attribute_rules.get(rule_name) )
                and rule['scriptExpression'] != t_rule['scriptExpression']
            ]
    return {k:v for k, v in _diffs.items() if v['added'] or v['removed'] or v['updated']}
        