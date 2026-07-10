from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from struct import Struct
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Self,
    TypedDict,
)

_HAS_ARCPY = True
_HAS_PANDAS = True
if TYPE_CHECKING:
    from arcpy import (
        Array,
        AsShape,
        Multipoint,
        Point,
        PointGeometry,
        Polygon,
        Polyline,
        SpatialReference,
    )
    from pandas import DataFrame
else:
    try:
        from pandas import DataFrame
    except ImportError:
        _HAS_PANDAS = False
        DataFrame = object
    try:
      from arcpy import (
          Array,
          AsShape,
          Multipoint,
          Point,
          PointGeometry,
          Polygon,
          Polyline,
          SpatialReference,
      )
    except ImportError:
        _HAS_ARCPY = False
        AsShape = None
        Point = None
        Multipoint = None
        PointGeometry = None
        SpatialReference = None
        Array = None
        Polyline = None
        Polygon = None


# All dates are anchored as float64 days from this date
START_DATE = datetime(1899, 12, 30, 0, 0, 0, 0)


class TablxHeader(TypedDict):
    version: int
    blocks: int
    byte_offset: int
    row_count: int
    _unknown: NotRequired[int]
    varying_section: NotRequired[int]


class TableHeader(TypedDict):
    version: int
    row_count: int
    row_size: int
    file_size: int
    fields_offset: int
    # _flag: int
    byte_offset: int
    has_deleted: NotRequired[bool]
    _unknown: NotRequired[int]


class MemoryReader:
    _struct_cache = dict[str, Struct]()
    __slots__ = 'byte_order', 'index', 'stack', 'view'

    def __init__(self, view: memoryview, start: int = 0,
                 *,
                 byte_order: Literal['@', '=', '<', '>', '!'] = '<',
                 stack_size: int = 10,
        ) -> None:
        self.view = view
        self.index = start
        self.byte_order = byte_order
        self.stack = deque[Any](maxlen=stack_size)

    def __len__(self) -> int:
        return self.view.nbytes - self.index

    @property
    def last(self) -> Any:
        return self.stack[-1]

    def scan(self, n: int) -> Self:
        self.index += n
        return self

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        if not any(fmt.startswith(c) for c in ('@', '=', '<', '>', '!')):
            fmt = self.byte_order + fmt
        struct = MemoryReader._struct_cache.setdefault(fmt, Struct(fmt))
        _sz = struct.size
        val = struct.unpack(self.view[self.index:self.index + _sz])
        self.scan(_sz)
        self.stack.extend(val)
        return val

    def int8(self, sign: bool = True) -> int:
        return self.unpack('b' if sign else 'B')[0]

    def int16(self, sign: bool = True) -> int:
        return self.unpack('h' if sign else 'H')[0]

    def int32(self, sign: bool = True) -> int:
        return self.unpack('i' if sign else 'I')[0]

    def int40(self, sign: bool = True) -> int:
        return self.int8(sign) | (self.int32(sign) << 8)

    def int48(self, sign: bool = True) -> int:
        return self.int8(sign) | (self.int40(sign) << 8)

    def int64(self, sign: bool = True) -> int:
        return self.unpack('q' if sign else 'Q')[0]

    def varint(self, sign: bool = True) -> int:
        neg = False
        if sign:
            val = self.int8(False)
            ret = (val & 0x3F)
            if val & 0x40:
                neg = True
            if not (val & 0x80):
                ret = -ret if neg else ret
                self.stack.append(ret)
                return ret
            shift = 6
        else:
            shift = ret = 0
        while True:
            val = self.int8(False)
            ret = ret | ((val & 0x7F) << shift)
            if not val & 0x80:
                break
            shift += 7
        ret = -ret if neg else ret
        self.stack.append(ret)
        return ret

    def float32(self) -> float:
        return self.unpack('f')[0]

    def float64(self) -> float:
        return self.unpack('d')[0]

    def read(self, length: int) -> bytes:
        val = self.view[self.index:self.index + length].tobytes()
        self.scan(length)
        return val

    def decode(self, length: int, encoding: str = 'utf-8') -> str:
        return self.read(length).decode(encoding)


class FieldReader(MemoryReader):

    def read_generic(self) -> dict[str, Any]:
        return {
            'width': self.int8(False),
            'flag': self.int8(False),
        }

    def read_default(self) -> dict[str, Any]:
        return {
            'width': self.int8(False),
            'flag': self.int8(False),
            'length': self.int8(False),
            'default': self.decode(self.last, 'utf-16le') if self.stack[-2] & 4 else None
        }

    def read_geom(self, *, has_m: bool = False, has_z: bool = False) -> dict[str, Any]:
        self.int8(False)  # discard this byte
        return {
            'flag': self.int8(False),
            'reference': self.decode(self.int16(), 'utf-16le'),
            'sys-flg': self.int8(False),
            'has-z': has_z,
            'has-m': has_m,
            'x-origin': self.float64(),
            'y-origin': self.float64(),
            'xy-scale': self.float64(),
            'm-origin': self.float64(),
            'm-scale': self.float64(),
            'z-origin': self.float64(),
            'z-scale': self.float64(),
            'xy-tol': self.float64(),
            'm-tol': self.float64(),
            'z-tol': self.float64(),
            'x-min': self.float64(),
            'y-min': self.float64(),
            'x-max': self.float64(),
            'y-max': self.float64(),
            'z-min': self.float64() if has_z else None,
            'z-max': self.float64() if has_z else None,
            'm-min': self.float64() if has_m else None,
            'm-max': self.float64() if has_m else None,
            'grids': self.unpack(f'{self.unpack('BI')[-1]}d'),
        }

    def read_string(self) -> dict[str, Any]:
        return {
            'max_len': self.int32(False),
            'flag': self.int8(False),
            'default': self.decode(self.varint(False), 'utf-8') if self.last & 4 else self.varint(False) and None
        }

    def read_raster(self) -> dict[str, Any]:
        return {
            '_unk_flg': self.int8(),
            'flag': self.int8(False),
            'col_type': self.decode(self.int8(False), 'utf-16le'),
            'wkt': self.decode(self.int8(False), 'utf-16le'),
            'flags': (flg := self.int8(False)),
            'xorigin': self.float64() if flg else None,
            'yorigin': self.float64() if flg else None,
            'xyscale': self.float64() if flg else None,
            'morigin': self.float64() if flg else None,
            'mscale': self.float64() if flg else None,
            'zorigin': self.float64() if flg else None,
            'zscale': self.float64() if flg else None,
            'xytol': self.float64() if flg else None,
            'mtol': self.float64() if flg else None,
            'ztol': self.float64() if flg else None,
            'raster_type': self.int8(),
        }


class GeometryReader(MemoryReader):
    def read_point(self, info: dict[str, Any], _as_shape: bool = False) -> Any:
        _xyscale: float = info['xy-scale']
        _mscale: float = info['m-scale']
        _zscale: float = info['z-scale']
        _xorigin: float = info['x-origin']
        _yorigin: float = info['y-origin']
        _zorigin: float = info['z-origin']
        _morigin: float = info['m-origin']
        _ref: str = info['reference']

        pt: dict[str, float] = {
            'x': (
                (self.varint(False) - 1) / _xyscale
                or float('nan')
            ) + _xorigin,
            'y': (
                (self.varint(False) - 1) / _xyscale
                or float('nan')
            ) + _yorigin,
            'z': (
                (self.varint(False) - 1) / _zscale
                or float('nan')
            ) + _zorigin if info['has-z'] else float('nan'),
            'm': (
                (self.varint(False) - 1) / _mscale
                or float('nan')
            ) + _morigin if info['has-m'] else float('nan'),
        }
        if _as_shape:
            return PointGeometry(
                Point(pt['x'], pt['y'], pt['z'], pt['m']),
                spatial_reference=SpatialReference(text=_ref)
            )
        return pt

    def read_multipoint(self, size: int, info: dict[str, Any], _as_shape: bool = False) -> Any:
        _xyscale: float = info['xy-scale']
        _xmin: float = info['x-min']
        _ymin: float = info['y-min']
        _mscale: float = info['m-scale']
        _zscale: float = info['z-scale']
        _hasz: bool = info['has-z']
        _hasm: bool = info['has-m']
        _xorigin: float = info['x-origin']
        _yorigin: float = info['y-origin']
        _zorigin: float = info['z-origin']
        _morigin: float = info['m-origin']
        _ref: str = info['reference']

        point_count = self.varint(False)
        bounds: dict[str, float] = {
            'xmin': self.varint(False) / _xyscale + _xorigin,
            'ymin': self.varint(False) / _xyscale + _yorigin,
            'xmax': self.varint(False) / _xyscale + _xmin,
            'ymax': self.varint(False) / _xyscale + _ymin,
        }
        points: list[dict[str, float]] = []
        dx = dy = dz = 0
        for _ in range(point_count):
            dx += self.varint()
            dy += self.varint()
            dx += self.varint() if _hasz else 0
            points.append(
                {
                    'x': dx / _xyscale + _xorigin,
                    'y': dy / _xyscale + _yorigin,
                    'z': (dz / _zscale + _zorigin) if _hasz else float('nan'),
                }
            )

        if _as_shape:
            return Multipoint(
                Array([Point(*p.values(), ID=i) for i, p in enumerate(points)]),
                spatial_reference=SpatialReference(text=_ref)
            )
        return points

    def _read_multi(self, info: dict[str, Any], _as_shape: bool = False) -> Any:
        _xyscale: float = info['xy-scale']
        _xmin: float = info['x-min']
        _ymin: float = info['y-min']
        _mscale: float = info['m-scale']
        _zscale: float = info['z-scale']
        _hasz: bool = info['has-z']
        _hasm: bool = info['has-m']
        _xorigin: float = info['x-origin']
        _yorigin: float = info['y-origin']
        _zorigin: float = info['z-origin']
        _morigin: float = info['m-origin']
        _ref: str = info['reference']

        point_count = self.varint(False)
        part_count = self.varint(False)
        bounds: dict[str, float] = {
            'xmin': self.varint(False) / _xyscale + _xorigin,
            'ymin': self.varint(False) / _xyscale + _yorigin,
        }
        bounds['xmax'] = self.varint(False) / _xyscale + bounds['xmin']
        bounds['ymax'] = self.varint(False) / _xyscale + bounds['ymin']

        part_point_count = [
            self.varint(False)
            for _ in range(part_count)
        ] if part_count > 1 else [point_count]

        rings: list[Any] = []
        for part in range(part_count):
            dx = dy = dz = 0
            points: list[Any] = []
            for point in range(part_point_count[part]):
                dx += self.varint()
                dy += self.varint()
                dx += self.varint() if _hasz else 0
                if _as_shape:
                    points.append(
                        Point(
                            dx / _xyscale + _xorigin,
                            dy / _xyscale + _yorigin,
                            (dz / _zscale + _zorigin) if _hasz else float('nan'),
                            ID=point,
                        )
                    )
                else:
                    points.append(
                        {
                            'x': dx / _xyscale + _xorigin,
                            'y': dy / _xyscale + _yorigin,
                            'z': (dz / _zscale + _zorigin) if _hasz else float('nan'),
                        }
                    )
            rings.append(points)
        return rings

    def read_polyline(self, info: dict[str, Any], _as_shape: bool = False) -> Any:
        points = self._read_multi(info, _as_shape)
        self.read(1)  # ?
        if _as_shape:
            return Polyline(Array(points), spatial_reference=SpatialReference(text=info['reference']))
        return points

    def read_polygon(self, info: dict[str, Any], _as_shape: bool = False) -> Any:
        points = self._read_multi(info, _as_shape)
        if _as_shape:
            return Polygon(Array(points), spatial_reference=SpatialReference(text=info['reference']))
        return points

    def read_general_polyline(self, size: int, info: dict[str, Any], _as_shape: bool = False) -> dict[str, Any]:
        self.scan(size)
        return {'polyline': 'TODO'}

    def read_general_polygon(self, size: int, info: dict[str, Any], _as_shape: bool = False) -> dict[str, Any]:
        self.scan(size)
        return {'polygon': 'TODO'}

    def read_general_point(self, size: int, info: dict[str, Any], _as_shape: bool = False) -> dict[str, Any]:
        self.scan(size)
        return {'general_point': 'TODO'}

    def read_general_multipoint(self, size: int, info: dict[str, Any], _as_shape: bool = False) -> dict[str, Any]:
        self.scan(size)
        return {'general_multipoint': 'TODO'}

    def read_general_multipatch(self, size: int, info: dict[str, Any], _as_shape: bool = False) -> dict[str, Any]:
        self.scan(size)
        return {'general_multipatch': 'TODO'}


class SimpleAnnotation(TypedDict):
    shape: Any
    text: str
    leaders: Any
    extra: NotRequired[list[Any]]


class CompoundAnnotation(SimpleAnnotation):
    type: str
    symbol: dict[str, Any]
    blendingMode: str
    attributes: dict[str, Any]
    placement: str


class AnnoReader(MemoryReader):

    def read_simple_anno(self, spatial_reference: SpatialReference | None = None) -> SimpleAnnotation:
        """Read a simple Annotation element"""
        strings: list[str] = []
        strings.append(self.decode(self.varint(False)))
        self.scan(1)  # 50? Possibly the leader point length?
        while self.index < self.view.nbytes:
            strings.append(self.decode(self.varint(False)))
            self.scan(12)  # Garbage buffer between strings

        data: SimpleAnnotation = {'shape': None, 'text': '', 'leaders': {}}
        if len(strings) >= 2:
            data['shape'] = [tuple(map(float, pt.split(','))) for pt in strings[0].split(';')]
            data['text'] = strings[1]
        if len(strings) >= 3:
            data['leaders'] = [json.loads(strings[2])] if strings[2] else []
        if len(strings) > 3:
            data['extra'] = strings[3:]

        if not _HAS_ARCPY:
            return data

        _points = Array(Point(*pt, ID=i) for i, pt in enumerate(data['shape']))
        data['shape'] = Polyline(_points, spatial_reference)
        if data.get('leaders'):
            data['leaders'] = [
                PointGeometry(Point(*leader.values()), spatial_reference)
                for leader in data['leaders']
            ]
        return data

    def read_compound_anno(self, spatial_reference: SpatialReference | None = None) -> CompoundAnnotation:
        """Read a compound Annotation element"""
        data = json.loads(self.decode(self.varint(False)))
        if not _HAS_ARCPY:
            return data

        data['shape'] = AsShape(data['shape'], esri_json=True).projectAs(spatial_reference or 'GCS_WGS_1984')
        for leader in data['leaders']:
            leader['point'] = PointGeometry(Point(*leader['point'].values()), spatial_reference)
        return data


class RowReader(MemoryReader):

    def read_geometry_field(self, info: dict[str, Any], _as_shape: bool = False) -> Any:
        size = self.varint(False)
        idx = self.index  # Store this since type is included in the blob
        geo_type = self.varint(False)
        size -= self.index - idx
        geo_reader = GeometryReader(view=self.view, start=self.index)
        self.scan(size)
        return {'SHAPE': geo_type}
        if geo_type in (1, 9, 21, 11):
            shape = geo_reader.read_point(info=info, _as_shape=_as_shape)
        elif geo_type in (8, 20, 28, 18):
            shape = geo_reader.read_general_multipoint(size, info, _as_shape=_as_shape)
        elif geo_type in (3, 10, 23, 13):
            shape = geo_reader.read_polyline(info, _as_shape=_as_shape)
        elif geo_type in (5, 19, 25, 15):
            shape = geo_reader.read_polygon(info, _as_shape=_as_shape)
        elif geo_type & 0xff == 50:
            shape = geo_reader.read_general_polyline(size, info, _as_shape=_as_shape)
        elif geo_type & 0xff == 51:
            shape = geo_reader.read_general_polygon(size, info, _as_shape=_as_shape)
        elif geo_type & 0xff == 52:
            shape = geo_reader.read_general_point(size, info, _as_shape=_as_shape)
        elif geo_type & 0xff == 53:
            shape = geo_reader.read_general_multipoint(size, info, _as_shape=_as_shape)
        elif geo_type & 0xff == 54:
            shape = geo_reader.read_general_multipatch(size, info, _as_shape=_as_shape)
        else:
            raise ValueError(f'Unknown geometry type {geo_type}')
        self.index = geo_reader.index
        return shape

    def read_binary_field(self) -> bytes:
        return self.read(self.varint(False))

    def read_string_field(self, encoding: str = 'utf-8') -> str:
        val = self.read(self.varint(False))
        try:
            return val.decode(encoding)
        except UnicodeDecodeError:
            return str(val)

    def read_xml_field(self, encoding: str = 'utf-8', strict: bool = False) -> str:
        val = self.read(self.varint(False))
        try:
            return val.decode(encoding, 'strict' if strict else 'replace')
        except UnicodeDecodeError:
            return str(val)

    def read_raster_field(self, raster_type: int) -> str | int | bytes:
        if raster_type == 0:
            return self.decode(self.varint(False), 'utf-16le')
        elif raster_type == 1:
            return self.int32()
        elif raster_type == 2:
            return self.read(self.varint(False))
        raise ValueError(f'Unknown Raster Type {raster_type}')

    def read_uuid_field(self) -> str:
        b = self.read(16)
        return (
            '{'
                f"{b[3]:02x}{b[2]:02x}{b[1]:02x}{b[0]:02x}-"
                f"{b[5]:02x}{b[4]:02x}-"
                f"{b[7]:02x}{b[6]:02x}-"
                f"{b[8]:02x}{b[9]:02x}-"
                f"{b[10]:02x}{b[11]:02x}{b[12]:02x}{b[13]:02x}{b[14]:02x}{b[15]:02x}"
            '}'
        )

    def read_objectid_field(self, version: int) -> int:
        if version == 3:
            return self.int32()
        elif version == 4:
            return self.int64()
        else:
            raise ValueError(f'Unknown OID version {version}')

    def read_generic_field(self, field_type: str) -> Any:
        if field_type == 'int16':
            return self.int16()
        elif field_type == 'int32':
            return self.int32()
        elif field_type == 'float32':
            return self.float32()
        elif field_type == 'float64':
            return self.float64()
        elif field_type == 'int64':
            return self.int64()
        elif field_type == 'datetime':
            return START_DATE + timedelta(self.float64())
        elif field_type == 'date':
            return (START_DATE + timedelta(self.float64())).date()
        elif field_type == 'time':
            return (START_DATE + timedelta(self.float64())).time()
        elif field_type == 'offsetdate':
            _date = (START_DATE + timedelta(self.float64()))
            _utc_offset = timedelta(hours=self.int16())
            return _date + _utc_offset


class FileGDB:
    """Input a file gdb and get access to the raw table definitions"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.gdb_catalog = GDBTable(path, 'a00000001', 'GDB_SystemCatalog')
        self.gdb_tune = GDBTable(path, 'a00000002', 'GDB_DBTune')
        self.gdb_refs = GDBTable(path, 'a00000003', 'GDB_SpatialRefs')
        self.gdb_items = GDBTable(path, 'a00000004', 'GDB_Items')
        self.gdb_item_types = GDBTable(path, 'a00000005', 'GDB_ItemTypes')
        self.gdb_item_relationships = GDBTable(path, 'a00000006', 'GDB_ItemRelationships')
        self.gdb_item_relationship_types = GDBTable(path, 'a00000007', 'GDB_ItemRelationshipTypes')

    @cached_property
    def index(self) -> dict[str, str]:
        return {
            tab['Name']: filename
            for tab in self.gdb_catalog
            if (filename := f'a{hex(tab['ID'])[2:].zfill(8)}')
            and (self.path / f'{filename}.gdbtable').exists()
        }

    @cached_property
    def tables(self) -> dict[str, GDBTable]:
        _index = {v: k for k, v in self.index.items()}
        return {
            display_name: GDBTable(self.path, table.stem, display_name)
            for table in self.path.glob('*.gdbtable')
            if (display_name := _index.get(table.stem, table.stem))
            and table.exists()
        }

    @property
    def type_map(self) -> dict[str, str]:
        """Mapping of internal feature type UUID to feature type name"""
        return {typ['UUID']: typ['Name'] for typ in self.gdb_item_types}

    @property
    def name_map(self) -> dict[str, str]:
        """Mapping of internal feature UUID to feature name"""
        return {itm['UUID']: itm['Name'] for itm in self.gdb_items}

    @property
    def dead_tables(self) -> dict[str, GDBTable]:
        return {k: v for k, v in self.tables.items() if k not in self.index}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(...{self.path.name}, tables={len(self.tables)})'

    def __iter__(self) -> Iterator[GDBTable]:
        yield from self.tables.values()

    def __getitem__(self, key: str) -> GDBTable:
        if key in self.tables:
            return self.tables[key]
        raise KeyError(f'Unable to find table {key} in {self.path.name}')


class GDBTable:
    """GDB Table File reader"""

    _as_shape = _HAS_ARCPY

    _geom_types = {
        1: 'point',
        2: 'multipoint',
        3: 'polyline',
        4: 'polygon',
        9: 'multipatch',
    }

    _field_types = {
        0:  'int16',
        1:  'int32',
        2:  'float32',
        3:  'float64',
        4:  'string',
        5:  'datetime',
        6:  'objectid',
        7:  'geometry',
        8:  'binary',
        9:  'raster',
        10: 'guid',
        11: 'globalid',
        12: 'xml',
        13: 'int64',
        14: 'date',
        15: 'time',
        16: 'offsetdate',
    }

    __slots__ = (
        '_displayname',
        '_row_offset',
        '_table_cache',
        '_table_file',
        '_table_time',
        '_tablx_cache',
        '_tablx_file',
        '_tablx_time',
        '_version',
        'path',
    )

    _version: int  # Literal[3,4]
    path: Path

    # Tablx file cache
    _tablx_file: Path
    _tablx_time: int
    _tablx_cache: bytes

    # Table file cache
    _table_file: Path
    _table_time: int
    _table_cache: bytes

    _row_offset: int  # Set when `.fields` is accessed and read
    _displayname: str

    def __init__(self, path: Path | str, name: str, display_name: str | None = None) -> None:
        self.path = Path(path)
        if (self.path / f'{name}.cdf').exists():
            raise ValueError(
                'Compressed Tables are not supported currently! '
                'Uncompress the database to use this module'
            )
        self._displayname = display_name or name
        self._tablx_file = self.path / f'{name}.gdbtablx'
        self._table_file = self.path / f'{name}.gdbtable'
        self._tablx_time = self._modified(self._tablx_file)
        self._tablx_cache = self._tablx_file.read_bytes()
        self._table_time = self._modified(self._table_file)
        self._table_cache = self._table_file.read_bytes()
        self._version: int = Struct('<i').unpack(self._tablx_cache[:4])[0]

    @property
    def name(self) -> str:
        return self._displayname

    def _modified(self, fl: Path) -> int:
        return fl.stat().st_mtime_ns

    def __repr__(self) -> str:
        return f'GDBTable({self._displayname}, {len(self)} rows)'

    def to_markdown(self, *,
                    max_col: int = 48,
                    truncate: int | None | Literal[True] = None,
                    align: Literal['left', 'center', 'right'] = 'left',
        ) -> str:
        rows: list[str] = []
        maxlen = max(len(str(v)) for r in self for v in r.values())
        maxlen = max_col if maxlen > max_col else maxlen
        headers = list(f'{k:^{maxlen}}' for k in self.fields.keys())
        if truncate is True:
            truncate = maxlen
        header_row = f'| {'|'.join(headers)} |'
        rows.append(header_row)
        row_sep = '-' * (len(header_row) // len(headers))
        header_sep = '|-' + '|'.join(row_sep[:-1] for _ in range(len(headers))) + '-|'
        rows.append(header_sep)
        spec = f'<{maxlen}'
        if align == 'center':
            spec = f'^{maxlen}'
        elif align == 'right':
            spec = f'>{maxlen}'
        for row in self:
            vals: list[str] = []
            for val in row.values():
                val = str(val).encode('unicode-escape').decode('utf-8')
                if truncate and len(val) > truncate:
                    val = val[:truncate][:-3] + '...'
                vals.append(f'{val!s:{spec}}')
            rows.append(f'| {'|'.join(vals)} |')
        return '\n'.join(rows)

    def to_csv(self, fl: Path | str, *, sep: str = ',', newline: str = '\n') -> Path:
        fl = Path(fl).with_suffix('.csv')
        headers = sep.join(self.fields.keys())
        rows = [sep.join(map(str, row.values())) for row in self]
        with fl.open('wt') as csvfile:
            csvfile.write(headers)
            csvfile.write(newline)
            csvfile.write(newline.join(rows))
            csvfile.write(newline)
        return fl

    def to_dict(self, fields: None | Sequence[str] = None) -> dict[str, list[Any]]:
        data = defaultdict[str, list[Any]](list)

        # Preserve field order
        if fields:
            for field in fields:
                data[field] = []

        for row in self:
            for k, v in row.items():
                if fields is None or k in fields:
                    data[k].append(v)
        return data

    def to_dataframe(self, fields: None | Sequence[str] = None) -> DataFrame:
        if not _HAS_PANDAS:
            raise SystemError('pandas is not available, install with `pip install pandas`')
        return DataFrame(self.to_dict(fields))

    @property
    def tablx(self) -> bytes:
        if (m_time := self._modified(self._tablx_file)) != self._tablx_time:
            self._tablx_time = m_time
            self._tablx_cache = self._tablx_file.read_bytes()
        return self._tablx_cache

    @property
    def table(self) -> bytes:
        if (m_time := self._modified(self._table_file)) != self._table_time:
            self._table_time = m_time
            self._table_cache = self._table_file.read_bytes()
        return self._table_cache

    @property
    def tablx_header(self) -> TablxHeader:
        reader = MemoryReader(
            memoryview(self.tablx),
            byte_order='<',
        )
        if self._version == 3:
            return {
                'version': reader.int32(False),
                'blocks': reader.int32(False),
                'row_count': reader.int32(False),
                'byte_offset': reader.int32(False),
            }
        elif self._version == 4:
            return {
                'version': reader.int32(False),
                'blocks': reader.int32(False),
                '_unknown': reader.int32(False),
                'byte_offset': reader.int32(False),

                # Special case for 64bit OIDs
                'row_count':
                    reader.scan(reader.last * 1024 * reader.stack[-3]).int64(False)
                    if reader.stack[-3] else 0,
                'varying_section':
                    reader.int32(False)
                    if reader.stack[-4] else 0
            }
        raise ValueError(f'Unknown GDB Version {self._version} (must be 3 or 4)')

    @property
    def table_header(self) -> TableHeader:
        reader = MemoryReader(
            memoryview(self.table),
            byte_order='<',
        )
        if self._version == 3:
            return {
                'version': reader.int32(False),
                'row_count': reader.int32(False),
                'row_size': reader.int32(False),
                'byte_offset': reader.int32(False),
                '_unknown': reader.int64(False),
                'file_size': reader.int64(False),
                'fields_offset': reader.int64(False),
            }
        elif self._version == 4:
            return {
                'version': reader.int32(False),
                'has_deleted': bool(reader.int32(False)),
                'row_size': reader.int32(False),
                'byte_offset': reader.int32(False),
                'row_count': reader.int64(False),
                'file_size': reader.int64(False),
                'fields_offset': reader.int64(False),
            }
        raise ValueError(f'Unknown GDB Version {self._version} (must be 3 or 4)')

    @property
    def field_descriptions(self) -> dict[str, Any]:
        reader = MemoryReader(
            memoryview(self.table),
            self.table_header['fields_offset'],
            byte_order='<'
        )
        return {
            'header_size': reader.int32(),
            'version': reader.int32(),
            'geom_info': reader.int32(False),
            'shape_type': self._geom_types.get(reader.last & 0xff, 'none'),
            'has_m': bool(reader.last >> 24 & (1 << 6)),
            'has_z': bool(reader.last >> 24 & (1 << 7)),
            'columns': reader.int16()
        }

    @property
    def field_offsets(self):
        reader = MemoryReader(
            memoryview(self.tablx),
            start=16,
            byte_order='<',
        )
        row_count = self.tablx_header['row_count']
        row_size = self.tablx_header['byte_offset']
        for _ in range(row_count):
            if row_size == 4:
                yield reader.int32(False)
            elif row_size == 5:
                yield reader.int40(False)
            elif row_size == 6:
                yield reader.int48(False)

    @property
    def fields(self) -> dict[str, dict[str, Any]]:
        _fields: dict[str, dict[str, Any]] = {}
        desc = self.field_descriptions
        has_m = desc['has_m']
        has_z = desc['has_z']
        cols = desc['columns']
        reader = FieldReader(
            memoryview(self.table),
            start=self.table_header['fields_offset'] + 14,
            byte_order='<',
        )
        for _ in range(cols):
            info: dict[str, Any] = {
                'name': reader.decode(reader.int8(False) * 2, 'utf-16le'),
                'alias': reader.decode(reader.int8(False) * 2, 'utf-16le'),
                'field_type': self._field_types.get(reader.int8(False), 'Unknown'),
                'flag': None,
            }
            field_type = info['field_type']
            if field_type == 'string':
                info.update(reader.read_string())
            elif field_type == 'geometry':
                info.update(reader.read_geom(has_m=has_m, has_z=has_z))
            elif field_type == 'raster':
                info.update(reader.read_raster())
            elif field_type in ('objectid', 'binary', 'guid', 'globalid', 'xml'):
                info.update(reader.read_generic())
            else:
                info.update(reader.read_default())
            _fields[info['name']] = info

            if info['flag'] is None:
                info['nullable'] = None
                info['required'] = None
                info['editable'] = None
            else:
                info['nullable'] = bool(info['flag'] & 1)
                info['required'] = bool(info['flag'] & 2)
                info['editable'] = bool(info['flag'] & 4)
        self._row_offset = reader.index
        return _fields

    def __len__(self) -> int:
        return self.table_header['row_count']

    def __iter__(self) -> Iterator[dict[str, Any]]:
        row_count = self.table_header['row_count']
        if not row_count: return
        fields = self.fields
        offset = self._row_offset
        nullable = sum(1 for f in fields.values() if f['nullable'])
        ver = self.tablx_header['version']
        yielded_rows = 0
        reader = RowReader(memoryview(self.table), start=offset)
        for fid, offset in enumerate(self.field_offsets, start=1):
            if not offset: continue
            if yielded_rows == row_count: break

            reader.index = offset
            _blob_len = reader.int32(False)  # unused?

            row: dict[str, Any] = {}
            flags: list[int] = [
                reader.int8(False)
                for _ in range(0, nullable, 8)
            ]
            flag_test = 0
            for name, field_info in fields.items():
                field_type = field_info['field_type']

                if field_type == 'objectid':
                    row[name] = fid
                    continue

                if flags and field_info['nullable']:
                    is_null = (flags[flag_test >> 3] & (1 << (flag_test % 8)))
                    flag_test += 1
                    if is_null:
                        row[name] = None
                        continue

                if field_type == 'geometry':
                    row[name] = reader.read_geometry_field(info=field_info, _as_shape=self._as_shape)
                elif field_type == 'binary':
                    row[name] = reader.read_binary_field()
                elif field_type == 'raster':
                    row[name] = reader.read_raster_field(field_info['raster_type'])
                elif field_type in ('string', 'xml'):
                    row[name] = reader.read_string_field()
                elif field_type in ('guid', 'globalid'):
                    row[name] = reader.read_uuid_field()
                elif field_type == 'objectid':
                    row[name] = reader.read_objectid_field(ver)
                else:
                    row[name] = reader.read_generic_field(field_info['field_type'])
            yield row
            yielded_rows += 1
