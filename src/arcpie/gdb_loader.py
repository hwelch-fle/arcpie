from __future__ import annotations

from itertools import batched
from pathlib import Path
from struct import Struct
from typing import Any, NotRequired, TypedDict, Literal

class TablxHeader(TypedDict):
    version: int
    blocks: int
    byte_offset: int
    row_count: NotRequired[int]
    _unknown: NotRequired[int]

class TableHeader(TypedDict):
    version: int
    row_count: int
    row_size: int
    file_size: int
    fields_offset: int
    _flag: int
    has_deleted: NotRequired[bool]
    _unknown: NotRequired[int]

class FileGDB:
    """Input a file gdb and get access to the raw table definitions"""
    
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.gdb_info = GDBTable(path, 'a00000004')
    
    # TODO: Memoize this
    @property
    def tables(self) -> dict[str, GDBTable]:
        return {
            table.stem: GDBTable(self.path, table.stem)
            for table in self.path.glob('*.gdbtable')
        }


class MemoryReader:
    _struct_cache: dict[str, Struct] = {}
    
    __slots__ = 'view', 'index', 'byte_order', 'stack'
    def __init__(self, view: memoryview, start: int = 0,
                 *,
                 byte_order: Literal['@','=','<','>','!'] = '=',
        ) -> None:
        self.view = view
        self.index = start
        self.byte_order = byte_order
        self.stack: list[Any] = []
    
    @property
    def last(self) -> Any:
        return self.stack[-1]
    
    def scan(self, n: int) -> None:
        self.index += n
    
    def unpack(self, fmt: str) -> tuple[Any, ...]:
        if not any(c in fmt for c in ('@','=','<','>','!')):
            fmt = self.byte_order+fmt
        struct = self._struct_cache.setdefault(fmt, Struct(fmt))
        val = struct.unpack(self.view[self.index:self.index+struct.size])
        self.scan(struct.size)
        self.stack.extend(val)
        return val

    def int8(self, sign: bool=True) -> int:
        return self.unpack('b' if sign else 'B')[0]

    def int16(self, sign: bool=True) -> int:
        return self.unpack('h' if sign else 'H')[0]

    def int32(self, sign: bool=True) -> int:
        return self.unpack('i' if sign else 'I')[0]

    def int64(self, sign: bool=True) -> int:
        return self.unpack('q' if sign else 'Q')[0]
    
    def varint(self, sign: bool=True) -> int:
        val = self.int8(False) if sign else 0xff
        ret = (val & 0x3F)*sign
        shift = 6*sign
        neg = sign and val & 0x40
        while True:
            if not val & 0x80: break
            val = self.int8(False)
            ret = ret | ((val & 0x7F) << shift)
            shift += 7
        return ret *-1 if neg else 1
    
    def float32(self) -> float:
        return self.unpack('f')[0]
    
    def float64(self) -> float:
        return self.unpack('d')[0]

    def decode(self, length: int, encoding: str = 'utf-8') -> str:
        val = self.view[self.index:self.index+length].tobytes().decode(encoding)
        self.scan(length)
        return val

class FieldReader(MemoryReader):

    def read_generic(self) -> dict[str, Any]:
        return {
            'width': self.int8(False),
            'flag': self.int8(False),
        }

    def read_default(self) -> dict[str, Any]:
        return {
            'width'  : self.int8(False),
            'flag'   : self.int8(False),
            'length' : self.int8(False),
            'default': self.decode(self.last, 'utf-8') if self.stack[-2] & 4 else None
        }

    def read_shape(self, *, has_m: bool = False, has_z: bool = False) -> dict[str, Any]:
        self.int8(False) # discard this byte
        return {
            'flag'   : self.int8(False),
            'wkt'    : self.decode(self.int16(), 'utf-16-le'),
            'sys_flg': self.int8(False),
            'has_z'  : has_z,
            'has_m'  : has_m,
            'xorigin': self.float64(),
            'yorigin': self.float64(),
            'xyscale': self.float64(),
            'morigin': self.float64(),
            'mscale' : self.float64(),
            'zorigin': self.float64(),
            'zscale' : self.float64(),
            'xytol'  : self.float64(),
            'mtol'   : self.float64(),
            'ztol'   : self.float64(),
            'xmin'   : self.float64(),
            'ymin'   : self.float64(),
            'xmax'   : self.float64(),
            'ymax'   : self.float64(),
            'zmin'   : self.float64() if has_z else None,
            'zmax'   : self.float64() if has_z else None,
            'mmin'   : self.float64() if has_m else None,
            'mmax'   : self.float64() if has_m else None,
            'grids'  : self.unpack('BI')[-1],
            'g_res'  : self.unpack(f'{self.last}d'),
        }

    def read_string(self) -> dict[str, Any]:
        return {
            'max_len': self.int32(), 
            'flag'   : self.int8(False), 
            'default': (self.varint(False), self.decode(self.last, 'utf-16-le') if self.stack[-1] & 4 else None)[-1]
        }

    def read_raster(self) -> dict[str, Any]:
        self.int8(False) # discard
        return {
            'flag'    : self.int8(False),
            'col_type': self.decode(self.int8(False), 'utf-16-le'),
            'wkt'     : self.decode(self.int8(False), 'utf-16-le'),
            'flags'   : (flg := self.int8(False)),
            'xorigin' : self.float64() if flg != 0 else None,
            'yorigin' : self.float64() if flg != 0 else None,
            'xyscale' : self.float64() if flg != 0 else None,
            'morigin' : self.float64() if flg != 0 else None,
            'mscale'  : self.float64() if flg != 0 else None,
            'zorigin' : self.float64() if flg != 0 else None,
            'zscale'  : self.float64() if flg != 0 else None,
            'xytol'   : self.float64() if flg != 0 else None,
            'mtol'    : self.float64() if flg != 0 else None,
            'ztol'    : self.float64() if flg != 0 else None,
            'type'    : self.int8(),
        }


class GDBTable:
    """GDB Table File reader"""
    
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
        10: 'GUID',
        11: 'GlobalID',
        12: 'XML',
        13: 'int64',
        14: 'DateOnly',
        15: 'TimeOnly',
        16: 'DateTimeWithOffset',
    }

    
    _v3_table_headers = (
        'version', 'row_count', 'row_size', '_flag', 
        '_unknown', 'file_size', 'fields_offset'
    )
    _v4_table_headers = (
        'version', 'has_deleted', 'row_size', '_flag', 
        'row_count', 'file_size', 'fields_offset'
    )
    _v3_tablx_headers = (
        'version', 'blocks', 'row_count', 'byte_offset'
    )
    _v4_tablx_headers = (
        'version', 'blocks', '_unknown', 'byte_offset'
    )
    
    __slots__ = (
        '_version', 'path',
        '_tablx_file', '_table_file',
        '_table_cache', '_tablx_cache',
        '_table_time', '_tablx_time', 
    )
    
    def __init__(self, path: Path | str, name: str) -> None:
        self.path = Path(path)
        self._tablx_file = self.path / f'{name}.gdbtablx'
        self._table_file = self.path / f'{name}.gdbtable'
        self._tablx_time = self._modified(self._tablx_file)
        self._tablx_cache = self._tablx_file.read_bytes()
        self._table_time = self._modified(self._table_file)
        self._table_cache = self._table_file.read_bytes()
        self._version: int = Struct('<i').unpack(self._tablx_cache[:4])[0]

    def _modified(self, fl: Path) -> int:
        return fl.stat().st_mtime_ns

    def _read_tablx(self) -> bytes:
        if (m_time := self._modified(self._tablx_file)) != self._tablx_time:
            self._tablx_time = m_time
            self._tablx_cache = self._tablx_file.read_bytes()
        return self._tablx_cache
    
    def _read_table(self) -> bytes:
        if (m_time := self._modified(self._table_file)) != self._table_time:
            self._table_time = m_time
            self._table_cache = self._table_file.read_bytes()
        return self._table_cache
    
    @property
    def tablx_header(self) -> TablxHeader:
        tablx = memoryview(self._read_tablx())
        header = Struct('<4i').unpack(tablx[:16])
        if self._version == 3:
            return dict(zip(self._v3_tablx_headers, header)) # type: ignore
        elif self._version == 4:
            return dict(zip(self._v4_tablx_headers, header)) # type: ignore
        else:
            raise ValueError(
                f'Unknown GDB Version {self._version} (must be 3 or 4)'
            )
    
    @property
    def table_header(self) -> TableHeader:
        table = memoryview(self._read_table())
        header = Struct('<4i3q').unpack(table[:40])
        if self._version == 3:
            return dict(zip(self._v3_table_headers, header)) # type: ignore
        elif self._version == 4:
            return dict(zip(self._v4_table_headers, header)) # type: ignore
        else:
            raise ValueError(
                f'Unknown GDB Version {self._version} (must be 3 or 4)'
            )

    @property
    def field_descriptions(self) -> dict[str, Any]:
        # Skip header
        table = memoryview(self._read_table()[self.table_header['fields_offset']:])
        (
            header_size, version, geom_info, columns
        ) = Struct('<2iIh').unpack(table[:14])
        
        geom_type = geom_info & 0xff
        flags = geom_info >> 24
        return {
            'shape_type': self._geom_types.get(geom_type, 'none'),
            'has_m': bool(flags & (1 << 6)),
            'has_z': bool(flags & (1 << 7)),
            'header_size': header_size,
            'version': version,
            'columns': columns
        }
    
    @property
    def fields(self):
        desc = self.field_descriptions
        offset = self.table_header['fields_offset']+14
        reader = FieldReader(
            memoryview(self._read_table()), 
            start=offset, 
            byte_order='<',
        )
        for _ in range(desc['columns']):
            info: dict[str, Any] = {
                'name': reader.decode(reader.int8(False)*2, 'utf-16-le'),
                'alias': reader.decode(reader.int8(False)*2, 'utf-16-le'),
                'field_type': reader.int8(False),
            }
            field_type = info['field_type']
            if field_type == 4:
                info.update(reader.read_string())
            elif field_type == 7:
                info.update(reader.read_shape(has_m=desc['has_m'], has_z=desc['has_z']))
            elif field_type == 9:
                info.update(reader.read_raster())
            elif field_type in (6, 10, 11, 12):
                info.update(reader.read_generic())
            else:
                info.update(reader.read_default())
            yield info

g = FileGDB(r"C:\Users\hwelch\Desktop\collection\populated_production.gdb")
f = g.tables['a00000032']