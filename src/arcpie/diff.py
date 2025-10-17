from __future__ import annotations

"""Module for diffing ArcGIS objects"""

from typing import Literal
from arcpie.database import Dataset
from arcpie.project import Project

Diff = dict[Literal['added', 'removed'], list[str]]

def feature_diff(source: Dataset, target: Dataset) -> Diff:
    """Get features that are added/removed from source/target"""
    _diff: Diff = {}
    _diff['added'] = list(set(target.feature_classes) - set(source.feature_classes))
    _diff['removed'] = list(set(source.feature_classes) - set(target.feature_classes))
    return _diff if _diff['added'] or _diff['removed'] else {}

def field_diff(source: Dataset, target: Dataset) -> dict[str, Diff]:
    """Get fields that are added/removed from source/target"""
    _diffs: dict[str, Diff] = {}
    for fc_name, source_fc in source.feature_classes.items():
        if (tarfc := target.feature_classes.get(fc_name, None)) is not None:
            _diffs.setdefault(fc_name, {})
            _diffs[fc_name]['added'] = [f for f in tarfc.fields if f not in source_fc.fields]
            _diffs[fc_name]['removed'] = [f for f in source_fc.fields if f not in tarfc.fields]
    return {k:v for k, v in _diffs.items() if v['added'] or v['removed']}

def layer_diff(source: Project, target: Project) -> dict[str, Diff]:
    """Get layers that are added/removed from source/target"""
    _diffs: dict[str, Diff] = {}
    for source_map in source.maps:
        map_name = source_map.name
        if (tarmap := target.maps.get(source_map.name, None)) is not None:
            _diffs.setdefault(map_name, {})
            _diffs[map_name]['added'] = list(set(tarmap.layers.names) - set(source_map.layers.names))
            _diffs[map_name]['removed'] = list(set(source_map.layers.names) - set(tarmap.layers.names))
    return {k:v for k, v in _diffs.items() if v['added'] or v['removed']}