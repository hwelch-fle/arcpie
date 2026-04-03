"""Module for internal utility functions to share between modules"""
from __future__ import annotations
import builtins
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
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
    SupportsIndex,
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
    Multipoint,
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
    
    Raises:
        ValueError if the point is disjoint from the line and `snap` is unset
    
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
    
    # Handle firstPoint
    if pt_meas < delta:
        pt_meas = delta
        point = line.positionAlongLine(pt_meas)
    # Handle lastPoint
    elif pt_meas == line.length:
        pt_meas = line.length - delta
        point = line.positionAlongLine(pt_meas)
        
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
        
        # nan checks
        if math.isnan(self.x):
            self.x = 0
        if math.isnan(self.y):
            self.y = 0
        if math.isnan(self.z):
            self.z = 0
        
        # Magnitude
        self.dist = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        
        # Secondary condition for null vector
        if self.dist == 0:
            self.is_null = True
        
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
        targ = self.tail_geom.move(
            dx=self.y*other.z - self.z*other.y,
            dy=self.z*other.x - self.x*other.z,
            dz=self.x*other.y - self.y*other.x,
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
        return self.x*other.x + self.y*other.y + self.z*other.z
    
    def __xor__(self, other: Vector):
        return self.dot(other)
    
    def angle(self, other: Vector) -> float:
        """Get the angle in radians between two vectors
        
        Note: When checking angle against a null vector, 0 is returned
        """
        if self.is_null or other.is_null:
            return 0
        
        other = Vector(self.tail, other >> self.tail)
        v = round((self^other)/(self.dist*other.dist), 15)
        ang = round(math.acos(v), 15)        
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
    
    
class PolylineEditor:
    """Allow simple polyline editing using indexes and slices
    
    multipart polylines will be indexed on global point index,
    ```
        [[p1: 3pts], [p2: 2pts]] ->  [0, 1, 2, 3, 4]
                                      ^^p1^^^|^p2^^
    ```
    """
    def __init__(self, polyline: Polyline) -> None:
        self._orig_polyline = polyline
        self.polyline = polyline
    
    @property
    def original(self) -> Polyline:
        """The original Polyline given to the PolylineEditor"""
        return self._orig_polyline
    
    @property
    def first_point(self) -> PointGeometry:
        """PointGeometry of the Polyline's firstPoint"""
        return PointGeometry(self.polyline.firstPoint, self.polyline.spatialReference)
    
    @first_point.setter
    def first_point(self, point: PointGeometry) -> None:
        """Update the first point of the polyline"""
        self[0] = point
    
    @property
    def last_point(self) -> PointGeometry:
        """PointGeometry of the Polyline's lastPoint"""
        return PointGeometry(self.polyline.lastPoint, self.polyline.spatialReference)
    
    @last_point.setter
    def last_point(self, point: PointGeometry) -> None:
        """Update the last point of the polyline"""
        self[-1] = point
    
    @property
    def centroid(self) -> PointGeometry:
        """Centroid of the feature if it is within the feature (same as measure@50%)"""
        return PointGeometry(self.polyline.centroid, self.polyline.spatialReference)
    
    @property
    def true_centroid(self) -> PointGeometry:
        """Center of gravity of the feature (not always in the line)"""
        return PointGeometry(self.polyline.trueCentroid, self.polyline.spatialReference)
    
    @property
    def parts(self) -> list[Polyline]:
        """A list of parts of the Polyline as Polylines (singlepart polylines will contain one part)"""
        return [
            Polyline(Array([p.centroid for p in part]), self.polyline.spatialReference) 
            for part in iter_parts(self.polyline)
        ]
    
    @parts.setter
    def parts(self, parts: list[Polyline]) -> None:
        """Update the geometry with a list of polyline parts"""
        self.polyline = self.merge_lines(parts)
    
    @property
    def part_editors(self) -> list[PolylineEditor]:
        """Get PolylineEditors for each part in the line"""
        return [PolylineEditor(p) for p in self.parts]
    
    @part_editors.setter
    def part_editors(self, parts: list[PolylineEditor]) -> None:
        """modify the part editors to update the polyline"""
        self.polyline = self.merge_lines(p.polyline for p in parts)
    
    @property
    def segments(self) -> list[Polyline]:
        """Get all 2-point segments that make up the line"""
        segs: list[Polyline] = []
        for part in self.part_editors:
            segs.extend(self.from_points(part[i:i+2]) for i in range(len(part)-1))
        return segs
    
    # Magic interfaces
    
    def __repr__(self) -> str:
        return f'PolylineEditor({repr(self.polyline)})'
    
    def __len__(self) -> int:
        return self.polyline.pointCount

    def __iter__(self) -> Iterator[PointGeometry]:
        return iter_points(self.polyline)
    
    def __contains__(self, item: Point | PointGeometry) -> bool:
        return not item.disjoint(self.polyline)
    
    @overload
    def __getitem__(self, key: slice) -> list[PointGeometry]: ...
    @overload
    def __getitem__(self, key: SupportsIndex) -> PointGeometry: ...
    def __getitem__(self, key: SupportsIndex | slice):
        _points = list(self)
        return _points[key]
    
    @overload
    def __setitem__(self, key: slice, value: Iterable[Point] | Iterable[PointGeometry]) -> None: ...
    @overload
    def __setitem__(self, key: SupportsIndex, value: Point | PointGeometry) -> None: ...
    def __setitem__(self, key: SupportsIndex | slice, value: Point | PointGeometry | Iterable[Point] | Iterable[PointGeometry]) -> None:
        _parts = self.part_editors
        if isinstance(value, (Point, PointGeometry)):
            value = self._cast_point(value)
        else:
            value = [self._cast_point(p) for p in value]

        if isinstance(key, slice) and isinstance(value, list):
            part_idx, local_idx = self._part_at_index(key.start or 0)
            local_stop = local_idx + (key.stop - key.start) if key.stop else None
            key = slice(local_idx, local_stop, key.step)
            _points = list(_parts[part_idx])
            _points[key] = value
            _parts[part_idx] = PolylineEditor(self.from_points(_points))
            
        elif isinstance(key, SupportsIndex) and isinstance(value, PointGeometry):
            part_idx, local_idx = self._part_at_index(key)
            _points = list(_parts[part_idx])
            _points[key] = value
            _parts[part_idx] = PolylineEditor(self.from_points(_points))
            
        else:
            raise ValueError(f'Unsupported types for key: {type(key)}, value: {type(value)}')
        
        self.polyline = self.merge_lines(p.polyline for p in _parts)
    
    @overload
    def get[D](self, key: slice, default: D = None) -> list[PointGeometry] | D: ...
    @overload
    def get[D](self, key: SupportsIndex, default: D = None) -> PointGeometry | D: ... 
    def get[D](self, key: SupportsIndex | slice, default: D = None ) -> PointGeometry | list[PointGeometry] | D:
        try:
            return self[key]
        except IndexError:
            return default
    
    # Polyline operator overloads
    
    def __add__(self, other: Polyline | PolylineEditor) -> PolylineEditor | None:
        if isinstance(other, PolylineEditor):
            other = other.polyline
        if self.polyline.disjoint(other):
            return None
        pl = self.polyline.intersect(other, 2)
        assert isinstance(pl, Polyline)
        return PolylineEditor(pl)
    
    def __or__(self, other: Polyline | PolylineEditor) -> PolylineEditor:
        if isinstance(other, PolylineEditor):
            other = other.polyline
        pl = self.polyline.union(other)
        assert isinstance(pl, Polyline)
        return PolylineEditor(pl)
    
    def __sub__(self, other: Polyline | PolylineEditor) -> PolylineEditor:
        if isinstance(other, PolylineEditor):
            other = other.polyline
        pl = self.polyline.difference(other)
        assert isinstance(pl, Polyline)
        return PolylineEditor(pl)
    
    def __xor__(self, other: Polyline | PolylineEditor) -> PolylineEditor:
        if isinstance(other, PolylineEditor):
            other = other.polyline
        pl = self.polyline.symmetricDifference(other)
        assert isinstance(pl, Polyline)
        return PolylineEditor(pl)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, PolylineEditor):
            return self.polyline == other.polyline
        elif isinstance(other, Polyline):
            return self.polyline == other
        return super().__eq__(other)
    
    def __ne__(self, other: object) -> bool:
        if isinstance(other, PolylineEditor):
            return self.polyline != other.polyline
        elif isinstance(other, Polyline):
            return self.polyline != other
        return super().__ne__(other)
    
    # List Interface
    
    def _cast_point(self, point: Point | PointGeometry) -> PointGeometry:
        return PointGeometry(point, self.polyline.spatialReference) if isinstance(point, Point) else point
        
    def _part_at_index(self, index: SupportsIndex) -> tuple[int, int]:
        """Internal method for mapping global index to local part index"""
        index = index.__index__()
        index = abs(index) - 1 if index < 0 else index
        for idx, part in enumerate(self.parts):
            if index > part.pointCount:
                index -= part.pointCount
            else:
                return idx, index
        raise IndexError('index out of range')
    
    def pop(self, index: SupportsIndex = -1, /) -> PointGeometry:
        """Remove a point from a polyline at the specified index (default: -1)"""
        part_idx, local_idx = self._part_at_index(index)
        
        _parts = list(self.parts)
        _new_part = list(self.part_editors[part_idx])
        pt = _new_part.pop(local_idx)
        
        _parts[part_idx] = self.from_points(_new_part)
        self.polyline = self.merge_lines(_parts)
        return pt
    
    def insert(self, index: SupportsIndex, point: Point | PointGeometry) -> None:
        """Insert a point at the specified index
        
        Note:
            Inserts the point at the global point index, whichever part that may be in
            If you want to insert a point at a specific part index, use `.parts` to access the 
            part and then insert the point there. This does not apply to single part features since 
            the local part index is equal to the global index.
        """
        point = self._cast_point(point)
        part_idx, local_idx = self._part_at_index(index)
        
        _parts = list(self.parts)
        _new_part = list(self.part_editors[part_idx])
        _new_part.insert(local_idx, point)
        
        _parts[part_idx] = self.from_points(_new_part)
        self.polyline = self.merge_lines(_parts)
    
    def index(self, point: Point | PointGeometry, start: SupportsIndex = 0, stop: SupportsIndex | None = None) -> int:
        """Get the global index of the first instance of the specified point, raise a ValueError if the point is not in the Polyline"""
        for idx, p in enumerate(list(self)[start:stop]):
            if not point.disjoint(p):
                return idx
        raise ValueError(f'Point {point} not found in polyline')
    
    def append(self, point: Point | PointGeometry) -> None:
        """Append a point to the last part of the Polyline"""
        _points = list(self.part_editors[-1])
        
        _points.append(self._cast_point(point))
        _parts = self.parts[:-1]
        
        _parts.append(self.from_points(_points))
        self.polyline = self.merge_lines(_parts)
        
    def extend(self, points: Iterable[Point | PointGeometry]) -> None:
        """Extend the last part of the polyline with the points"""
        _points = list(self.part_editors[-1])
        
        _points.extend(self._cast_point(p) for p in points)
        _parts = self.parts[:-1]
        
        _parts.append(self.from_points(_points))
        self.polyline = self.merge_lines(_parts)
    
    def reverse(self) -> None:
        """Reverse the polyline parts and the points of each part of the polyline
        
        Example:
            [[a, b], [c, d]] -> [[d, c], [b, a]]
            [a, b, c, d] -> [d, c, b, a]
        """
        #self.polyline = self.merge_lines(self.from_points(reversed(list(part))) for part in reversed(self.part_editors))    
        rev = self.polyline.reverseOrientation() # type: ignore
        assert isinstance(rev, Polyline)
        self.polyline = rev
    
    def count(self, point: Point | PointGeometry) -> int:
        """Get the count of points in the line that match the input point"""
        point = self._cast_point(point)
        return list(self).count(point)
    
    def copy(self) -> PolylineEditor:
        """Return a copy of the current editor with the active polyline state as the original polyline"""
        return PolylineEditor(self.polyline)
    
    def remove(self, point: Point | PointGeometry) -> None:
        """Remove the fist occurrence of the point from the polyline"""
        self.pop(self.index(point))
        
    def discard(self, point: Point | PointGeometry) -> None:
        """Silently remove points without raising an error if it doesn't exist"""
        try:
            self.remove(point)
        except IndexError:
            return
    
    # Special Operations
    
    def dedupe(self, keep: Literal['first', 'last'] = 'last') -> int:
        """Remove all duplicate points in a line keeping last instance (returns the number of points removed)
        
        Note:
            If a duplciate is found in a seperate part, it is not considered a duplicate and will be kept 
            setting `keep` to `'last'` will remove duplicates from the end of the line first
        """
        _parts = list(self.part_editors)
        _removed = 0
        for part in _parts:
            _points = list(part)
            if keep == 'first': _points.reverse()
            
            for point in filter(lambda point: _points.count(point) > 1, _points):
                _points.remove(point)
                _removed += 1
            
            if keep == 'first': _points.reverse()
            part.polyline = self.from_points(_points)
        
        self.polyline = self.merge_lines(p.polyline for p in _parts)
        return _removed
    
    def reset(self) -> None:
        """Revert all changes to `polyline` and restore the original geometry"""
        self.polyline = self._orig_polyline
    
    def snap(self, line: Polyline | PolylineEditor) -> None:
        if isinstance(line, PolylineEditor):
            line = line.polyline
        _parts = self.part_editors
        for i, part in enumerate(_parts):
            _parts[i].polyline = PolylineEditor.from_points(line.snapToLine(p) for p in part)
        self.polyline = self.merge_lines(p.polyline for p in _parts)
    
    def generalize(self, distance: float) -> None:
        """Generalize the polyline"""
        self.polyline = self.polyline.generalize(distance)
    
    def append_part(self, part: Polyline | PolylineEditor) -> None:
        """Add a part to the polyline"""
        _parts = self.part_editors
        if isinstance(part, PolylineEditor):
            new_parts = part.part_editors
        else:
            new_parts = PolylineEditor(part).part_editors
        _parts.extend(new_parts)
        self.polyline = self.merge_lines(p.polyline for p in _parts)
    
    def pop_part(self, index: SupportsIndex = -1, /) -> Polyline:
        """pop a part from the polyline (default: -1)"""
        _parts = self.part_editors
        if len(_parts) == 1:
            raise ValueError('cannot pop parts from a single part polyline')
        part = _parts.pop(index)
        self.polyline = self.merge_lines(p.polyline for p in _parts)
        return part.polyline
    
    def move(self, vec: Vector) -> None:
        """Move the polyline along a Vector"""
        _parts = self.part_editors
        for i, part in enumerate(_parts):
            _parts[i].polyline = self.from_points(vec.translate(p) for p in part)
        self.polyline = self.merge_lines(p.polyline for p in _parts)
    
    def project_as(self, ref: SpatialReference | int) -> None:
        """Project the polyline in the given reference"""
        if isinstance(ref, int):
            ref = SpatialReference(ref)
        self.polyline = self.polyline.projectAs(ref)
    
    def orenent_with(self, line: Polyline | PolylineEditor) -> None:
        """Alter the direction of the polyline to match the direction of the input line (only works for lines that overlap)"""
        if isinstance(line, PolylineEditor):
            line = line.polyline
        
        if not line.disjoint(self.polyline) and (int_part := line.intersect(self.polyline, 2)):
            if self.polyline.measureOnLine(int_part.firstPoint) > self.polyline.measureOnLine(int_part.lastPoint):
                self.reverse()
        
    def align(self, line: Polyline | PolylineEditor, *, tolerance: float = 0.01) -> None:
        """Adjust points of the polyline to be coincident with the target line of they are within `tolerance` units 
        If multiple points are within the tolerance, the closest is picked
        """
        if isinstance(line, Polyline):
            line = PolylineEditor(line)
        _other_points = list(line)
        for idx, point in enumerate(self):
            nearest = min(_other_points, key=lambda p: p.distanceTo(point))
            if nearest.distanceTo(point) <= tolerance:
                self[idx] = nearest
    
    def split_at_angle(self, ang: float, units: Literal['degrees', 'radians'] = 'radians', *, tolerance: float = 0.1) -> list[Polyline]:
        """Split the polyline at points where the instantaneous angle (radians) is greater than the specified angle (radians)
        
        Args:
            ang: The target angle at a point to be split at
            units: Specify the units of the provided angle and tolerance
            tolerance: +/- of target angle to trigger a split (default: `0.1rad`/`~5.7deg`)
        """
        ang = abs(ang)
        tolerance = abs(tolerance)
        if units == 'degrees':
            ang = abs(math.radians(ang)%math.pi)
            tolerance = abs(math.radians(tolerance))
        
        segs: list[Polyline] = []
        for part in self.part_editors:
            last_split = 0
            for idx, point in enumerate(part[1:-1], start=1):
                l, r = vectors_at(part.polyline, point)
                if abs(l@r) < ang + tolerance:
                    segs.append(self.from_points(part[last_split:idx+1]))
                    last_split = idx
                
            if last_split != len(part)-1:
                segs.append(self.from_points(part[last_split:]))
        return segs
    
    def intersections(self, other: Polyline | PolylineEditor) -> Iterator[PointGeometry]:
        """Iterable of Point Intersections between this line and the other
        
        Note: Intersections are de-duplicated and returned as PointGeometry objects
        """
        if isinstance(other, PolylineEditor):
            other = other.polyline
        
        intersection = self.polyline.intersect(other, 1)
        if isinstance(intersection, PointGeometry) and intersection.isMultipart:
            intersection = Multipoint(Array([p for p in intersection]), self.polyline.spatialReference)
            
        if isinstance(intersection, Multipoint):
            seen: list[PointGeometry] = []
            for p in (PointGeometry(p, self.polyline.spatialReference) for p in intersection):
                if not any(p == s for s in seen):
                    yield p
                    seen.append(p)
        else:
            yield PointGeometry(intersection.centroid, self.polyline.spatialReference)
    
    # Constructors
    
    @classmethod
    def merge_lines(cls, lines: Iterable[Polyline]) -> Polyline:
        """Merge a sequence of Polylines into one Polyline (uses `union` so each line becomes a part)
        
        Example:
            ```python
                lines = [[a,b,c,d], [e,f,g,h]]
                new = PolylineEditor.merge_lines(lines)
                new == Polyline([p1[a,b,c,d], p2[e,f,g,h]])
            ```
        """
        # Merge algorithm that keeps sequential touching lines in the same part, 
        # but allows for disjoint parts to remain disjoint
        _parts: list[list[PointGeometry]] = []
        _last_point: PointGeometry | None = None
        _ref: SpatialReference | None = None
        for line in lines:
            if _ref is None:
                _ref = line.spatialReference
            points = list(cls(line))
            if _last_point and not points[0].disjoint(_last_point):
                _parts[-1].extend(points)
            else:
                _parts.append(points)
            _last_point = points[-1]
        return Polyline(
            Array(
                Array(p.centroid for p in part) 
                for part in _parts
            ), 
            _ref
        )
        #return reduce(lambda acc, l: acc.union(l), lines) # type: ignore
    
    @classmethod
    def from_points(cls, points: Iterable[Point | PointGeometry], ref: SpatialReference | None = None) -> Polyline:
        points = [
            PointGeometry(p if isinstance(p, Point) else p.centroid, ref)
            for p in points
        ]
        return Polyline(Array([p.centroid for p in points]), ref)
        
