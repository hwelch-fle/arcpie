"""Module for diffing ArcGIS objects"""
from __future__ import annotations

from typing import Any, Literal

from arcpie.database import Dataset
from arcpie.project import Project

Diff = dict[Literal['added', 'removed', 'updated'], list[str] | list[dict[str, Any]]]


def feature_diff(source: Dataset, target: Dataset) -> Diff:
    """Get features that are added/removed from source/target

    Args:
        source (Dataset): The starting point of the delta
        target (Dataset): The ending point of the delta

    Returns:
        (Diff): FeatureClass delta
    """
    diff: Diff = {}
    diff['added'] = list(set(target.feature_classes) - set(source.feature_classes))
    diff['removed'] = list(set(source.feature_classes) - set(target.feature_classes))
    diff['updated'] = [
        fc for fc in source.feature_classes
        if fc in target
        and (
            set(target.feature_classes[fc].fields) != set(source.feature_classes[fc].fields)
            or not target.feature_classes[fc].py_types.values() == source.feature_classes[fc].py_types.values()
        )
    ]
    return diff


def field_diff(source: Dataset, target: Dataset) -> dict[str, Diff]:
    """Get fields that are added/removed from source/target

    Args:
        source (Dataset): The starting point of the delta
        target (Dataset): The ending point of the delta

    Returns:
        (dict[str, Diff]): A Mapping of feature names to field deltas
    """
    diffs: dict[str, Diff] = {}
    for fc_name, source_fc in source.feature_classes.items():
        if (target_fc := target.feature_classes.get(fc_name, None)) is not None:
            diffs.setdefault(fc_name, {})
            diffs[fc_name]['added'] = [f for f in target_fc.fields if f not in source_fc.fields]
            diffs[fc_name]['removed'] = [f for f in source_fc.fields if f not in target_fc.fields]

            # Compare fields from matching features
            to_compare = ('baseName', 'aliasName', 'defaultValue', 'domain',
                           'editable', 'isNullable', 'length', 'name',
                           'precision', 'required', 'scale', 'type')
            source_fields = {f.baseName: f for f in source_fc.describe.fields}
            target_fields = {f.baseName: f for f in target_fc.describe.fields}
            diffs[fc_name]['updated'] = [
                {f: changes}
                for f in source_fields
                if f in target_fields
                and (changes := {
                    attr: f"{getattr(source_fields[f], attr)} -> {getattr(target_fields[f], attr)}"
                    for attr in to_compare
                    if getattr(source_fields[f], attr) != getattr(target_fields[f], attr)
                })
            ]

    return {k: v for k, v in diffs.items() if v['added'] or v['removed'] or v['updated']}


def layer_diff(source: Project, target: Project) -> dict[str, Diff]:
    """Get layers that are added/removed from source/target

    Args:
        source (Project): The starting point of the delta
        target (Project): The ending point of the delta

    Returns:
        (dict[str, Diff]): A Mapping of map names to layer deltas
    """
    diffs: dict[str, Diff] = {}
    for source_map in source.maps:
        map_name = source_map.unique_name
        if (target_map := target.maps.get(source_map.unique_name, None)) is not None:
            diffs.setdefault(map_name, {})
            diffs[map_name]['added'] = list(set(target_map.layers.names) - set(source_map.layers.names))
            diffs[map_name]['removed'] = list(set(source_map.layers.names) - set(target_map.layers.names))
            diffs[map_name]['updated'] = [
                lay.unique_name for lay in source_map.layers
                if hasattr(lay, 'name')
                and lay.unique_name in target_map.layers.names
                and (source_cim := lay.cim_dict)
                and (target_cim := target_map.layers[lay.unique_name].cim_dict)
                and any(
                    str(source_cim.get(k, 'Source')) != str(target_cim.get(k, 'Target'))
                    for k in ['renderer', 'labelClasses']  # Only compare symbology and labels
                )
            ]
    return {k: v for k, v in diffs.items() if v['added'] or v['removed'] or v['updated']}


def table_diff(source: Project, target: Project) -> dict[str, Diff]:
    """Get tables that are added/removed/changed from source/target

    Args:
        source (Project): The starting point of the delta
        target (Project): The ending point of the delta

    Returns:
        (dict[str, Diff]): A Mapping of map names to table deltas
    """
    diffs: dict[str, Diff] = {}
    for source_map in source.maps:
        map_name = source_map.unique_name
        if (target_map := target.maps.get(source_map.unique_name, None)) is not None:
            diffs.setdefault(map_name, {})
            diffs[map_name]['added'] = list(set(target_map.tables.names) - set(source_map.tables.names))
            diffs[map_name]['removed'] = list(set(source_map.tables.names) - set(target_map.tables.names))
            diffs[map_name]['updated'] = [
                t.unique_name for t in source_map.tables
                if hasattr(t, 'name')
                and t.unique_name in target_map.tables.names
                and (source_cim := t.cim_dict)
                and (target_cim := target_map.tables[t.unique_name].cim_dict)
                and any(
                    str(source_cim.get(k, 'Source')) != str(target_cim.get(k, 'Target'))
                    for k in ['displayField']  # Only compare display field
                )
            ]
    return {k: v for k, v in diffs.items() if v['added'] or v['removed'] or v['updated']}


def attribute_rule_diff(source: Dataset, target: Dataset) -> dict[str, Diff]:
    """Get a diff of rules for matching FeatureClasses in the dataset

    Args:
        source (Dataset): The starting point of the delta
        target (Dataset): The ending point of the delta

    Returns:
        (dict[str, diff]): A Mapping of Features to rules with a delta type
    """

    diffs: dict[str, Diff] = {}
    for fc_name, source_fc in source.feature_classes.items():
        if (target_fc := target.feature_classes.get(fc_name, None)) is not None:
            diffs.setdefault(fc_name, {})
            diffs[fc_name]['added'] = [rule for rule in target_fc.attribute_rules.names if rule not in source_fc.attribute_rules.names]
            diffs[fc_name]['removed'] = [rule for rule in source_fc.attribute_rules.names if rule not in target_fc.attribute_rules.names]
            diffs[fc_name]['updated'] = [
                rule_name
                for rule_name, rule in source_fc.attribute_rules.rules.items()
                if rule and (t_rule := target_fc.attribute_rules.get(rule_name))
                and rule['scriptExpression'] != t_rule['scriptExpression']
            ]
    return {k: v for k, v in diffs.items() if v['added'] or v['removed'] or v['updated']}


def domain_diff(source: Dataset, target: Dataset) -> Diff:
    """Get a diff of rules for matching FeatureClasses in the dataset

    Args:
        source (Dataset): The starting point of the delta
        target (Dataset): The ending point of the delta

    Returns:
        (Diff): A domain diff
    """
    diff: Diff = {}
    to_check: list[str] = ['codedValues', 'description', 'domainType', 'mergePolicy', 'splitPolicy', 'type']
    diff['added'] = [d.name for d in target.domains if d not in source.domains]
    diff['removed'] = [d.name for d in source.domains if d not in target.domains]
    diff['updated'] = [
        {source_domain.name: changes}
        for source_domain in source.domains
        if source_domain.name in target.domains
        and (changes := {
            attr: f"{getattr(source_domain, attr, None)} -> {getattr(target.domains[source_domain.name], attr, None)}"
            for attr in to_check
            if str(getattr(source_domain, attr, None)) != str(getattr(target.domains[source_domain.name], attr, None))
        })
    ]
    return diff
