from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, time

# TODO: Change this to annotationlib when 3.15 is adopted
from typing import Any, Literal, TypedDict, ForwardRef, get_type_hints

from arcpy import (
    Geometry,
    PointGeometry,
    Polygon,
    Polyline,
    Multipoint,
    Multipatch,
)

from ..featureclass import FeatureClass, Table
from ..cursor import FieldType, Field

__all__ = (
    
    # Field Annotations
    'FA_Type', 
    'FA_Precision', 
    'FA_Scale', 
    'FA_Length', 
    'FA_Alias', 
    'FA_Nullable', 
    'FA_Required',
    'FA_Domain',
    'FA_Default',
    
    # Token bases
    'GeometryShape',
    'PolygonShape',
    'PointShape',
    'PolylineShape',
    'MultiPointShape',
    'MultiPatchShape',
    'OIDToken',
)

# Consumable Annotations for use with building out schemas
# e.g. 
# 
# class MyFeature(TypedDict):
#     OBJECTID: Annotated[
#         int, # Py_Type
#         FA_Type('BIGINTEGER'),
#         FA_Alias('MyFeatureID'),
#     ]
#     Name: Annotated[
#         str,
#         FA_Length(50),
#         FA_Required(), # Flags default to True if included, False if not included
#         FA_Alias('Feature Name'),
#     ]
#     Type: Annotated[
#         str,
#         FA_Length(100),
#         FA_Required(False), # Explicitly state NotRequired
#         FA_Domain('MyFeatureDomain')
#     ]

FIELD_TYPE_MAP: dict[FieldType, type] = {
    'SHORT': int,
    'LONG': int,
    'BIGINTEGER': int,
    'FLOAT': float,
    'DOUBLE': float,
    'DATE': datetime,
    'DATEHIGHPRECISION': datetime,
    'TIMEONLY': time,
    'DATEONLY': datetime,
    'BLOB': bytes,
    'GUID': str,
    'RASTER': bytes,
    'SHORT': int,
    'TEXT': str,
}

class FieldAnnotation:
    __slots__ = ()

@dataclass(frozen=True, slots=True)
class FA_Type(FieldAnnotation):
    field_type: FieldType

@dataclass(frozen=True, slots=True)
class FA_Precision(FieldAnnotation):
    field_precision: int

@dataclass(frozen=True, slots=True)
class FA_Scale(FieldAnnotation):
    field_scale: int

@dataclass(frozen=True, slots=True)
class FA_Length(FieldAnnotation):
    field_length: int
    
@dataclass(frozen=True, slots=True)
class FA_Alias(FieldAnnotation):
    field_alias: str

@dataclass(frozen=True, slots=True)
class FA_Nullable(FieldAnnotation):
    field_is_nullable: bool = True

@dataclass(frozen=True, slots=True)
class FA_Required(FieldAnnotation):
    field_is_required: bool = True

@dataclass(frozen=True, slots=True)
class FA_Domain[T: str](FieldAnnotation):
    """Accepts a TypeVar Literal with valid domains"""
    field_domain: T

@dataclass(frozen=True, slots=True)
class FA_Default(FieldAnnotation):
    field_default: Any


# These bases can't use a Generic since the key @ required the special 
# functional constructor for TypedDict

# This is a fallback for any undefined geometry types
GeometryShape = TypedDict('GeometryShape', {'SHAPE@': Geometry})

PolygonShape = TypedDict('PolygonShape', {'SHAPE@': Polygon})
PointShape = TypedDict('PointShape', {'SHAPE@': PointGeometry})
PolylineShape = TypedDict('PolylineShape', {'SHAPE@': Polyline})
MultiPointShape = TypedDict('MultiPointShape', {'SHAPE@': Multipoint})
MultiPatchShape = TypedDict('MultiPatchShape', {'SHAPE@': Multipatch})

OIDToken = TypedDict('OIDToken', {'OID@': int})


# TODO: Add other token types? Like @CREATED or SHAPE@XY ?


# Any internal schemas need to be included here 
# so they can be filtered out when parsing a schema module
INTERNAL_SCHEMAS = (
    'GeometryShape',
    'PolygonShape',
    'PointShape',
    'PolylineShape',
    'MultiPointShape',
    'MultiPatchShape',
    'OIDToken',
)


type SchemaDocs = dict[str, str]
"""A mapping of fieldname -> field doc"""


# NOTE: .format(doc=docstring) this with a module docstring 
# any method/function using this should default the doc to 
# the Dataset name if none is provided
SCHEMA_IMPORTS = '''"""
{}
"""


from typing import TypedDict, Annotated
from datetime import datetime, time
from arcpie.schemas.fields import *

# If a domain module is included, these will be 
# added before the schema to allow domain linking
# in fields
try:
    import domains
except ImportError:
    domains = None
    
DOMAINS = domains and domains.DOMAINS

TYPE_CHECKING = False
if TYPE_CHECKING:
    # This allows autocompletion 
    # if you have a generated domains module
    assert domains
    DomainNames = domains.DomainNames
    FA_Domain = FA_Domain[DomainName]
    
else:
    DomainNames = None

'''


# Internal function for adding a docstring to all fields that 
# can be parsed by language servers like pyright
def _default_field_doc(f_def: Field) -> str:
    lines: list[str] = []
    for k, v in f_def.items():
        lines.append(rf"{k.replace('field_', '')}: {v}\n\n")
    return '"""' + ''.join(lines) + '"""'


def yield_schema(fc: FeatureClass[Any, Any] | Table[Any],
                 *, 
                 fallback_type: type = object, 
                 docs: SchemaDocs | None = None,
                 include_shape_token: bool = True,
                 include_oid_token: bool = True,
                 default_doc: Callable[[Field], str]=_default_field_doc,
    ) -> Iterator[str]:
    """Yield the code for a FeatureClass schema
    
    Args:
        fc: The FeatureClass/Table to generate a schema dict for
        fallback_type: The default type annotation for any fields that aren't mapped properly
        docs: Optional docs to include for each field (e.g. `{'FieldName': 'field doc', ...}`)
        include_shape_token: Include a `SHAPE@` key with the FeatureClass shape type (no effect on Tables)
        include_oid_token: Include the `OID@` key
        default_doc: A function that takes a Field dictionary and retuens a formatted doc (default: `k: v\n\n...`)
        
    Note:
        The schema type will be added as a __doc__ attribute to the definition
    """
    if not docs:
        docs = {}
        
    bases = ['TypedDict']
    if include_oid_token:
        bases.append('OIDToken')
        docs['OID@'] = '"""OID Token for `arcpy.da` Cursors"""'
        
    if include_shape_token and isinstance(fc, FeatureClass):
        _shape = fc.describe.shapeType
        if _shape == 'Polygon':
            bases.append('PolygonShape')
        elif _shape == 'Point':
            bases.append('PointShape')
        elif _shape == 'Polyline':
            bases.append('PolylineShape')
        elif _shape == 'Multipoint':
            bases.append('MultiPointShape')
        elif _shape == 'MultiPatch':
            bases.append('MultiPatchShape')
        else:
            bases.append('GeometryShape')
        
        docs['SHAPE@'] = f'"""Shape Token for `arcpy.da` Cursors: {_shape.__class__.__name__}"""'
    
    if len(bases) > 1:
        # Don't bother inheriting TypedDict directly if additional 
        # schema tokens are inherited
        bases = bases[1:]
    
    yield f"class {fc.name}({', '.join(bases)}):"
    yield f'    """{getattr(fc.describe, 'featureType', 'Table')}"""'
    
    # Sort the defs by name to make schema diffing more reliable
    for f_name, f_def in sorted(fc.field_defs.items(), key=lambda fd: fd[0]):
        
        f_type = f_def.get('field_type')
        f_pytype = fallback_type
        if f_type:
            f_pytype = FIELD_TYPE_MAP.get(f_type, fallback_type)
        f_length = f_def.get('field_length')
        f_precision = f_def.get('field_precision')
        f_scale = f_def.get('field_scale')
        f_alias = f_def.get('field_alias')
        if f_alias == f_name:
            # Only set an alias if it's different
            f_alias = None
        
        # By default, fields are nullable and required
        f_is_nullable = f_def.get('field_is_nullable', True)
        f_is_required = f_def.get('field_is_required', True)
        f_domain = f_def.get('field_domain')
        f_default = f_def.get('field_default')
        if isinstance(f_default, str):
            f_default = repr(f_default)
            
        yield f"    {f_name}: Annotated[{f_pytype.__name__},"
        if f_type:
            yield f"        FA_Type({repr(f_type)}),"
        if f_alias:
            yield f"        FA_Alias({repr(f_alias)}),"
        if f_default:
            yield f"        FA_Default({f_default}),"
        if f_domain:
            yield f"        FA_Domain({repr(f_domain)}),"
        if f_length:
            yield f"        FA_Length({f_length}),"
        if f_precision:
            yield f"        FA_Precision({f_precision}),"
        if f_scale:
            yield f"        FA_Scale({f_scale}),"
        if f_is_nullable == 'NULLABLE':
            yield f"        FA_Nullable(),"
        if f_is_required == 'REQUIRED':
            yield f"        FA_Required(),"
        yield "    ]"
        if docs:
            f_doc: str = docs.get(f_name) or default_doc(f_def)
        else:
            f_doc: str = default_doc(f_def)
        if f_doc: # Don't yield an empty string
            yield f'    {f_doc}'
        yield ""


# Table schemas are stored as TypedDict
type TableDef = Mapping[str, type]
type RootDef = Mapping[str, ForwardRef]

# Valid arguments to CreateFeatureClass for the Shape option
GeoType = Literal["POINT", "MULTIPOINT", "POLYGON", "POLYLINE", "MULTIPATCH"]


def parse_hierarchy(root: type | ForwardRef, skip_annos: bool = True) -> dict[str, Any]:
    """Parse the root schema and resolve all forward references to Table definitions"""
    _parsed: dict[str, Any] = {}
    root_types = get_type_hints(root, include_extras=True)
    for item, item_type in root_types.items():
        if (item_type.__doc__ or '').startswith('Feature Dataset'):
            _parsed[item] = parse_hierarchy(item_type)
        if not (item_type.__doc__ or '').startswith('Feature Dataset'):
            if (item_type.__doc__ or '').startswith('Annotation') and skip_annos:
                continue
            _parsed[item] = parse_fields(item_type)
    return _parsed

def parse_fields(table_def: TableDef) -> tuple[GeoType | None, dict[str, Field]]:
    """Parse a table definition generated by `yield_schema`
    
    Args:
        table_def: The table TypedDict with Annotated keys to parse
    
    Returns:
        A tuple containing None or the Feature Shape flag, and a mapping of fieldnames to field constructor args 
        
        e.g. `('POLYLINE', {'Field1': {'field_type': 'TEXT', ...}, ...})`
        
    Note:
        `None` as the first return value indicates a Table or a shapeless featureclass (`SHAPE@` required in the Schema)
    """
    geo_type: GeoType | None = None
    annos = get_type_hints(table_def, include_extras=True)
    fields: dict[str, Field] = {}
    for f_name, f_detail in annos.items():
        # Resolve ForwardRef and make sure to include Annotated __metadata__
        if '@' in f_name:
            if f_name == 'SHAPE@':
                # Get the shape type
                geo_type = str(f_detail.__name__).upper() # type: ignore (Shape name is capital case)
                if geo_type == 'POINTGEOMETRY': # type: ignore (Edge Case)
                    geo_type = 'POINT'
            continue
        
        # Consume annotations to build FieldDef
        field_opts: Field = {
            prop.__slots__[0]: getattr(prop, prop.__slots__[0])
            for prop in f_detail.__metadata__ # type: ignore (Annotated access)
        }
        fields[f_name] = field_opts
        
    return (geo_type, fields)
    
