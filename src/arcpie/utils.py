"""Module for internal utility functions to share between modules"""
from __future__ import annotations
import builtins
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import reduce
from io import BytesIO
import math
from pathlib import Path
import json

from tempfile import TemporaryDirectory
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
)

from networkx import (
    Graph, 
    shortest_path as nx_shortest_path, # type: ignore
    all_shortest_paths as nx_all_shortest_paths,
    NodeNotFound,
    NetworkXNoPath,
)

from arcpy import (
    AddMessage,
    AddWarning,
    AddError,
    Array,
    AsShape,
    Point,
    PointGeometry,
    Polygon,
    Polyline,
    SpatialReference,
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

def get_subtype_count(fc: Table | FeatureClass, drop_empty: bool=False) -> dict[str, int]:
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

if TYPE_CHECKING:
    from featureclass import _GeometryType, _Schema # type: ignore
def subtype_summary(fc: Table[_Schema] | FeatureClass[_GeometryType, _Schema], summary: Callable[[Iterator[_Schema]], Any]) -> dict[str, Any]:
    return {
        subtype['Name']: res
        for code, subtype in fc.subtypes.items() 
        if fc.subtype_field # has Subtypes
        and (res := summary(fc[where(f'{fc.subtype_field} = {code}')]))
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

from .schema import SchemaWorkspace, SchemaDataset

def convert_schema(schema: Dataset[Any]|Path|str, to: Literal['JSON', 'XLSX', 'HTML', 'PDF', 'XML', 'DYNAMIC_HTML']='JSON') -> BytesIO:
    """Convert a Schema from one format to another
    
    Args:
        schema (Dataset|Path|str): Path to the schemafile or Dataset to convert
        to (Literal['JSON', 'XLSX', 'HTML', 'PDF', 'XML', 'DYNAMIC_HTML']): Target format (default: 'JSON')
        
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
                
def split_lines_at_points(lines: Polyline | Sequence[Polyline] | Iterator[Polyline], points: Sequence[PointGeometry] | Iterator[PointGeometry]) -> Iterator[Polyline]:
    """Split a Polyline or Sequence/Iterable of polylines at provided points
    
    Args:
        lines (Polyline | Sequence[Polyline] | Iterator[Polyline]): The line or lines to split
        points (Sequence[PointGeometry] | Iterator[PointGeometry]): The points to split at
    
    Yields:
        (Polyline): Segments of the polyline split at the input points
    """
    if isinstance(lines, Polyline):
        lines = [lines]
    if not isinstance(points, list):
        points = list(points)
    
    for line in lines:
        int_points = [p for p in points if not p.disjoint(line)]
        if not int_points:
            yield line
            continue
        if all(p.touches(line) for p in int_points):
            yield line
            continue
        prev_measure = 0.0
        measures = sorted(line.measureOnLine(p) for p in int_points)
        for measure in measures + [line.length]:
            if prev_measure == measure:
                continue
            yield line.segmentAlongLine(prev_measure, measure).projectAs(line.spatialReference)
            prev_measure = measure
        

PointLike = PointGeometry | Point
LineCollection = FeatureClass[Polyline, Any] | Sequence[Polyline] | Iterator[Polyline]
@overload
def shortest_path( # type: ignore (all_paths changes return type)
    source: PointLike, 
    target: PointLike, 
    network: LineCollection, 
    *,
    all_paths: Literal[False]=False,
    method: Literal['dijkstra', 'bellman-ford']='dijkstra',
    weighted: bool=True,
    precision: int=6) -> Polyline | None: ...

@overload
def shortest_path(
    source: PointLike, 
    target: PointLike, 
    network: LineCollection, 
    *,
    all_paths: Literal[True]=True,
    method: Literal['dijkstra', 'bellman-ford']='dijkstra',
    weighted: bool=True,
    precision: int=6) -> list[Polyline] | None: ...

def shortest_path(
    source: PointLike, 
    target: PointLike, 
    network: LineCollection,
    *,
    all_paths: bool=False,
    method: Literal['dijkstra', 'bellman-ford']='dijkstra',
    weighted: bool=True,
    precision: int=6,
    ) -> Polyline | list[Polyline] | None:
    """Find the shortest path or paths given a source point, target point and network of Polylines
    
    Args:
        source (PointGeometry | Point): The start point for the path
        target (PointGeometry | Point): The end point for the path
        network (FeatureClass[Polyline, Any] | Sequence[Polyline] | Iterator[Polyline]): The polylines to traverse
        all_paths (bool): If True, yield all shortest paths from (default: False)
        method (Literal['dijkstra', 'bellman-ford']): The graph traversal algorithm to use (default: 'dijkstra')
        weighted (bool): Use line lengths to weight the paths (default: True)
        precision (int): Number of decimal places to round coordinates to (default: 6)
    
    Returns:
        (Polyline | None): The unioned polyline of the path or None if no path is found
    
    Yields:
        (Polyline): Yields all shortest paths if `all_paths` is set (None is still *returned* if no path found)
    
    Raises:
        (ValueError): When input arguments are not of the correct types (`PointLike`, `PointLike`, `LineCollection`)
        
    Example:
        ```python
        path = shortest_path(p1, p2, line_features)
        paths = shortest_path(p1, p2, line_features, all_paths=True)
        
        if path is None:
            print('No Path')
        else:
            print(path.length)
        
        # Check that path(s) were found
        if paths is None:
            print('No Path')
        else:
            for p in paths:
                print(p.length)
        ```   
    """
    
    # Parameter Validations
    if isinstance(network, FeatureClass):
        if network.describe.shapeType != 'Polyline':
            raise ValueError(f'network must have polyline geometry')
        network = list(network.shapes)
    
    if not isinstance(network, Sequence):
        network = list(network)
    
    if not network:
        return None
    
    if not isinstance(network[0], Polyline): # type: ignore
        raise ValueError(f'network must have polyline geometry')
    
    # Project point geometries to match reference of network, extract X,Y tuples
    if isinstance(source, PointGeometry):
        if source.isMultipart:
            raise ValueError('source point must not be multipart')
        source = source.projectAs(network[0].spatialReference).centroid
    _source = (round(source.X, precision), round(source.Y, precision))
    
    if isinstance(target, PointGeometry):
        if target.isMultipart:
            raise ValueError('target point must not be multipart')
        target = target.projectAs(network[0].spatialReference).centroid
    _target = (round(target.X, precision), round(target.Y, precision))
    
    G = Graph(
        [
            (
                (round(l.firstPoint.X, precision), round(l.firstPoint.Y, precision)), 
                (round(l.lastPoint.X, precision), round(l.lastPoint.Y, precision)), 
                {'shape': l, 'length': l.length}
            ) 
            for l in network
        ]
    )
    try:
        if all_paths:
            paths = nx_all_shortest_paths(G, _source, _target, weight='length' if weighted else None, method=method)
        else:
            paths = nx_shortest_path(G, _source, _target, weight='length' if weighted else None, method=method)
    except (NodeNotFound, NetworkXNoPath):
        return None

    if all_paths:
        _paths: list[Polyline] = []
        for path in paths:
            assert isinstance(path, list)
            edges: list[Polyline] = [G.get_edge_data(u, v)['shape'] for u, v in zip(path, path[1:])]
            _paths.append(reduce(lambda acc, s: acc.union(s), edges))  # pyright: ignore[reportArgumentType]
        return _paths
    
    else:
        assert isinstance(paths, list)
        edges: list[Polyline] = [G.get_edge_data(u, v)['shape'] for u, v in zip(paths, paths[1:])]
        if edges:
            return reduce(lambda acc, s: acc.union(s), edges) # pyright: ignore[reportReturnType]

def box_on_point(
    center: Point | PointGeometry, 
    width: float, height: float, 
    angle: float=0.0, 
    ref: SpatialReference|None=None, 
    start: Literal['tl', 'tr', 'bl', 'br']='tl'
    ) -> Polygon:
    """Build a rectangular box on a point
    
    Args:
        center (Point|PointGeometry): The center point of the box
        width (float): The width of the box
        height (float): The height of the box
        angle (float): An angle to roatate the box by in radians (default: 0.0)
        ref (SpatialReference|None): An optional spatial reference to apply to the output polygon
        start (Literal['tl', 'tr', 'bl', 'br']): The corner of the box that should be the start point (default: 'tl')
        
    Returns:
        (Polygon): A rectangular polygon (with provided ref or ref inhereted from center)
    """
    if isinstance(center, PointGeometry):
        ref = ref or center.spatialReference
        if center.spatialReference != ref:
            center = center.projectAs(ref)
        center = center.centroid
        
    h_width = width/2
    h_height = height/2
    tl = Point(center.X-h_width, center.Y+h_height)
    tr = Point(center.X+h_width, center.Y+h_height)
    bl = Point(center.X-h_width, center.Y-h_height)
    br = Point(center.X+h_width, center.Y-h_height)
    points = deque([tl, tr, br, bl])
    
    if start == 'tr':
        points.rotate(-1)
    elif start == 'br':
        points.rotate(-2)
    elif start == 'bl':
        points.rotate(-3)
        
    box = Polygon(Array(points), spatial_reference=ref)
    if angle:
        box = box.rotate(center, angle) # type: ignore
        assert isinstance(box, Polygon)
    return box

def two_point_circle(center: Point|PointGeometry, end: Point|PointGeometry, ref: SpatialReference|None=None) -> Polyline:
    """Create a circle using a center point and an end point
    
    Args:
        center (Point|PointGeometry): The center of the circle
        end: (Point|PointGeometry): The end point of the arc (distance from center is Circle radius)
        ref: (SpatialReference|None): The SpatialReference to use with the returned geometry
        
    Returns:
        (Polyline): A Circular Polyline
    
    Note:
        If PointGeometries are provided, they will be projected as the provided ref
        If no ref is provided, the shape will inherit the reference of the center
        
        Reference resolution is as follows:
        `ref -> center.spatialReference -> end.spatialReference`
        
        If both points are Point objects with no spatial reference, and no ref is provided, 
        The returned Polyline will have no spatial reference
    """
    if isinstance(center, PointGeometry):
        _c_ref = center.spatialReference
        if ref and _c_ref != ref:
            center = center.projectAs(ref)
        else:
            ref = ref or _c_ref
        center = center.centroid
    if isinstance(end, PointGeometry):
        _e_ref = end.spatialReference
        if ref and _e_ref != ref:
            end = end.projectAs(ref)
        else:
            ref = ref or _e_ref
        end = end.centroid
    
    esri_json: dict[str, Any] = {}
    _arc: dict[str, Any] = {'a': [[end.X, end.Y, end.Z, end.M], [center.X, center.Y], 0, 1]}
    esri_json['curvePaths'] = [[[end.X, end.Y, end.Z], _arc]]
    if ref:
        esri_json['spatialReference'] = {'wkid': ref.factoryCode, 'latestWkid': ref.factoryCode}
    return AsShape(esri_json, esri_json=True) # type: ignore

def center_circle(center: Point|PointGeometry, radius: float, ref: SpatialReference|None=None) -> Polyline:
    """Create a circle using a center point and a radius
    
    Args:
        center (Point|PointGeometry): The center of the circle
        radius: (float): The dist
        ref: (SpatialReference|None): The SpatialReference to use with the returned geometry
        
    Returns:
        (Polyline): A Circular Polyline
    
    Note:
        If a PointGeometry are provided, it will be projected as the provided ref
        If no ref is provided, the shape will inherit the reference of the center
        
        Reference resolution is as follows:
        `ref -> center.spatialReference`
        
        If no center reference can be found and no ref is provided, the returned geometry will 
        have no reference
    """
    if isinstance(center, PointGeometry):
        ref = ref or center.spatialReference
        center = center.centroid 
    return two_point_circle(center, Point(center.X, center.Y+radius), ref)


def vectors_at(line: Polyline, point: PointGeometry | Point, *, delta: float = 0.01, snap: bool = False) -> tuple[Vector, Vector]:
    """Get the vector of the line at the given point (p-delta -> p+delta)
    
    Args:
        line: The line to get a Vector for
        point: The PointGeometry specifying the vector location (projects to line reference)
        delta: The distance (meters) to traverse the line (default: `0.01`)
        snap: If the input point is disjoint from the line, snap it to the line (default: 'False')
    
    Returns:
        A tuple of Vectors representing the bearing to start and end at distance delta from point
        
    Note:
        If the in point is a start/end point on the line, one of the Vectors will be a null vector!
    """
    ref = line.spatialReference
    mpu = ref.metersPerUnit
    if isinstance(point, Point):
        point = PointGeometry(point, ref)
    else:
        point.projectAs(ref)
    
    if line.disjoint(point):
        if snap:
            point = line.snapToLine(point)
        else:
            raise ValueError('Vector point must not be disjoint from line (use `snap=True` to snap point to line)')
        
    pt_meas = line.measureOnLine(point)*mpu
    plus = line.positionAlongLine((pt_meas + delta)/mpu)
    minus = line.positionAlongLine((pt_meas - delta)/mpu)
    
    return Vector(point, plus), Vector(point, minus)


def vector_at(line: Polyline, point: PointGeometry | Point, *, delta: float = 0.01, snap: bool = False) -> Vector:
    """Get the vector of the line at the given point (p-delta -> p+delta)
    
    Args:
        line: The line to get a Vector for
        point: The PointGeometry specifying the vector location (projects to line reference)
        delta: The distance (meters) to traverse the line in both directions (default: `0.01`)
        snap: If the input point is disjoint from the line, snap it to the line (default: 'False')
    
    Returns:
        A Vector of length delta*2 that describes the slope of the line at the provided point
    """
    v1, v2 = vectors_at(line, point, delta=delta, snap=snap)
    return v1 + v2


def iter_points(line: Polyline, start: bool = True, end: bool = True) -> Iterator[PointGeometry]:
    """Get a point iterator for a Polyline
    
    Args:
        line: The Polyline to iterate points for
        start: Include the line startpoint (default: `True`)
        end: Include the line endpoint (default: `True`)
    Yields:
        PointGeometries for all points in the line
    """
    section = slice(0 if start else 1, None if end else -1)
    yield from (
        PointGeometry(point, line.spatialReference)
        for part in line
        for point in list[Point](part)[section] # type: ignore
    )


def iter_parts(line: Polyline, start: bool = True, end: bool = True) -> Iterator[Iterator[PointGeometry]]:
    """Get a part iterator for a Polyline
    
    Args:
        line: The Polyline to iterate parts for
        start: Include the line startpoint (default: `True`)
        end: Include the line endpoint (default: `True`)
    
    Yields:
        Iterators of part PointGeometries for all parts in the line
    """
    yield from (
        iter_points(Polyline(part, line.spatialReference, start, end)) 
        for part in line
    )


type Scalar = int | float


@dataclass
class Vector:
    """Simple Vector implementation that takes a start and end point (uses Spherical notation for theta and phi).\n
    
    If `PointGeometries` are passed as the head and tail points, the head
    will inherit the reference of the tail
    
    Attributes:
        x (float): The X component of the vector
        y (float): The Y component of the vector
        z (float): The Z component of the vector
        theta (float): The angle between the vector and the x-axis
        phi (float)L The angle between the vector and the z-axis
        dist (float): The magnitute of the vector (distance b/w start and end)
        mid (Point): The midpoint of the vector along its magnitude
        ref (SpatialReference | None): Ano optional spatial reference for all Vector geometry to inherit
        
    Methods:
        translate(point, mag): Translate a point along the Vector (Vector tail is set to point location)
        reverse: Reverse the Vector
        add(other): Vector addition (LHS is origin)
        subtract(other): Vector subtraction (LHS is origin)
        dot(other): Dot product of two vectors (LHS is origin)
        cross(other): Cross product of two vectors (LHS is origin)
        scale(scalar): Scale a vector using a scalar
        angle(other, order[`accute`|`obtuse`]=`accute`): Get angle between two vectors
        
        __rshift__: Implements `>>` for use in translating points (vector must be LHS)
        __neg__: Implements unary `-` operator for Vectors. Same as .reverse()
        __add__: Implements `+` operator for Vectors. RHS Vector will be used as anchor
        __sub__: Implements `-` operator for Vectors. RHS Vector will be used as anchor
        __xor__: Implements `^` operator for Vectors (dot product).
        __mul__: Implements `*` operator for Vectors. RHS can be Vector (cross product) or Scalar
        __matmul__: Implements `@` to determine *accute* angle between two vectors
    """
    #__slots__ = 'head', 'tail', 'ref', 'x', 'y', 'z', 'dist', 'theta', 'phi', 'mid', 'is_null', 
    
    tail: Point | PointGeometry
    head: Point | PointGeometry
    ref: SpatialReference | None = None
    
    def __post_init__(self) -> None:
        
        if isinstance(self.tail, PointGeometry):
            _ref = self.ref or self.tail.spatialReference
            self.tail = self.tail.projectAs(_ref).centroid
            self.ref = _ref
        
        if isinstance(self.head, PointGeometry):
            _ref = self.ref or self.head.spatialReference
            self.head = self.head.projectAs(_ref).centroid
            self.ref = _ref
        
        self.is_null = (
            self.head.X == self.tail.X
            and self.head.Y == self.tail.Y
            and self.head.Z == self.tail.Z
        )
        
        # Normalized Components
        self.x = self.head.X - self.tail.X
        self.y = self.head.Y - self.tail.Y
        self.z = (self.head.Z or 0) - (self.tail.Z or 0)
        
        # Magnitude
        self.dist = math.sqrt(self.x**2 + self.y**2 + self.z**2) if not self.is_null else 0
        
        # Angle components
        self.theta = math.atan2(self.y, self.x) if not self.is_null else 0
        self.phi = math.acos(self.z/self.dist) if not self.is_null else 0
        
        # Midpoint
        self.mid = self.translate(self.tail, self.dist/2) if not self.is_null else self.tail
    
    @property
    def polyline(self) -> Polyline:
        _ref = None
        start = self.tail
        end = self.head
        if isinstance(start, PointGeometry):
            _ref = start.spatialReference
            start = start.centroid
        if isinstance(end, PointGeometry):
            _ref = _ref or end.spatialReference
            end = end.centroid
        return Polyline(Array([start, end]), _ref)
    
    @property
    def head_geom(self) -> PointGeometry:
        """Get a point geometry object for the vector head (end)"""
        return self.head if isinstance(self.head, PointGeometry) else PointGeometry(self.head)
    
    @property
    def tail_geom(self) -> PointGeometry:
        """Get a point geometry object for the vector tail (start)"""
        return self.tail if isinstance(self.tail, PointGeometry) else PointGeometry(self.tail)
    
    @property
    def mid_geom(self) -> PointGeometry:
        """Get a point geometry object for the vector midpoint (inherits reference from head/tail)"""
        return self.mid if isinstance(self.mid, PointGeometry) else PointGeometry(self.mid)
    
    def __repr__(self) -> str:
        return f'Vector(x={self.x}, y={self.y}, z={self.z}, tail={self.tail})'
    
    def reverse(self) -> Vector:
        """Reversed vector"""
        return Vector(self.head, self.tail)
    
    def __neg__(self) -> Vector:
        return self.reverse()
    
    def norm(self) -> Vector:
        """Normal vector originating at tail"""
        return Vector(self.tail, self.translate(self.tail, 1))
    
    def __abs__(self):
        return self.norm()
    
    def add(self, other: Vector) -> Vector:
        """Vector addition originating at LHS (`+`)"""
        return Vector(self.tail, other >> self.head)
        
    def __add__(self, other: Vector):
        return self.add(other)
    
    def subtract(self, other: Vector) -> Vector:
        """Vector subtraction originating at LHS (`-`)"""
        return self + (-other)
    
    def __sub__(self, other: Vector):
        return self.subtract(other)
    
    def cross(self, other: Vector) -> Vector:
        """Cross product of two vectors originating at LHS (`*`)"""
        if self.is_null:
            return self
        if other.is_null: # Null vector at own tail
            return Vector(self.tail, self.tail)
        other = Vector(self.tail, other >> self.tail)
        targ = Point(
            X = self.y*other.z - self.z*other.y,
            Y = self.z*other.x - self.x*other.z,
            Z = self.x*other.y - self.y*other.x,
        )
        return Vector(self.tail, targ)
    
    def scale(self, scale: Scalar) -> Vector:
        """Scalar multiplication of vector (`*`)
         
        Note: Scaling by Zero will return a null vector located at the vector tail
        """
        if scale == 0: # Special case for creating a spatially aware null vector
            return Vector(self.tail, self.tail)
        return Vector(self.tail, self.translate(self.tail, scale*self.dist))
    
    def __mul__(self, other: Vector | Scalar) -> Vector:
        if isinstance(other, Vector):
            return self.cross(other)
        return self.scale(other)
    
    def __rmul__(self, other: Vector | Scalar) -> Vector:
        if isinstance(other, Vector):
            return other.cross(self)
        return self.scale(other)
    
    def dot(self, other: Vector) -> float:
        """Dot product of two vectors originating at LHS (`@`)"""
        other = Vector(self.tail, other >> self.tail)
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def __xor__(self, other: Vector):
        return self.dot(other)
    
    def angle(self, other: Vector, order: Literal['accute', 'obtuse'] = 'accute') -> float:
        """Get the angle in radians between two vectors (order can be `accute` or `obtuse`)
        
        Note: When checking angle against a null vector, 0 is returned
        """
        if self.is_null or other.is_null:
            return 0
        
        other = Vector(self.tail, other >> self.tail)
        ang = round(math.acos((self^other)/(self.dist*other.dist)), 15)
        rad = round(math.pi/2, 15)
        if ang == 0:
            return ang
        if ang > rad and order == 'accute':
            return round(ang - rad, 15)
        elif ang < rad and order == 'obtuse':
            return round(ang + rad, 15)
        return ang
    
    def __matmul__(self, other: Vector) -> float:
        return self.angle(other)
    
    def __lt__(self, other: Vector) -> bool:
        return self.dist < other.dist
    
    def __gt__(self, other: Vector) -> bool:
        return self.dist > other.dist
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vector):
            other = Vector(self.tail, other >> self.tail)
            return self^other == self.dist**2
        return super().__eq__(other)
    
    def __len__(self) -> float:
        return self.dist
    
    def __bool__(self) -> bool:
        return self.is_null
    
    @overload
    def translate(self, point: Point, mag: float | None = None) -> Point: ...
    @overload
    def translate(self, point: PointGeometry, mag: float | None = None) -> PointGeometry: ...
    def translate(self, point: Any, mag: float | None = None) -> Any:
        """Translate the provided point along the vector direction. `>>`
        
        The Point will be moved from its original location along the vector angle the provided distance.
        The location of the `Vector` object is not taken into account, only angle and magnitude
        
        Args:
            point: The point to translate along the given vector
            mag: The distance to translate the point (default: `self.mag`)
        
        Returns:
            A translated Point/PointGeometry
            
        Example:
            ```python
                trans_point = vec >> point
            ```
        """
        
        if self.is_null or mag == 0:
            return point
        
        if not isinstance(point, (Point, PointGeometry)):
            raise TypeError(f'point must be Point or PointGeometry not {type(point)}')
        
        target = point
        ref = None
        
        if isinstance(target, PointGeometry):
            ref = target.spatialReference
            target = target.centroid
        
        mag = mag or self.dist
        target = Point(
            target.X + (mag*self.x)/self.dist, 
            target.Y + (mag*self.y)/self.dist, 
            target.Z + (mag*self.z)/self.dist if target.Z is not None else None, 
            target.M, 
            target.ID,
        )
        
        return (
            PointGeometry(
                target, 
                ref, 
                has_z = target.Z is not None, 
                has_m = target.M is not None, 
                has_id = bool(target.ID), 
            ) 
            if isinstance(point, PointGeometry)
            else target
        )
    
    @overload
    def __rshift__(self, point: Point) -> Point: ...
    @overload
    def __rshift__(self, point: PointGeometry) -> PointGeometry: ...
    def __rshift__(self, point: Point | PointGeometry) -> Point | PointGeometry:
        return self.translate(point)
    