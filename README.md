# arcpie
A simple interface for working with arcpy Featureclasses

## The Old
```python
import arcpy

fc1 = r'C:\Data\db.gdb\FC1'
fc2 = r'C:\Data\db.gdb\FC2'
l1 = arcpy.management.MakeFeatureLayer(fc1, 'memory/fc1', 'size > 10')
l2 = arcpy.management.MakeFeatureLayer(fc2, 'memory/fc2')
arcpy.management.SelectLayerByLocation(l2, fc1)
print(arcpy.management.getCount(l2)[0])
```

## The New
```python
from arcpie import FeatureClass, count

fc1 = FeatureClass[Polyline](r'C:\Data\db.gdb\FC1')
fc2 = FeatureClass[PointGeometry](r'C:\Data\db.gdb\FC2')

with fc1.where('length > 10'):
    print(count(fc2[fc1.footprint()])

def under(dist: int) -> Callable:
    def _inner(row: dict[str, Any]) -> bool:
        return row['SHAPE@'].length < dist
    return _inner

for row in fc1[under(10)]:
    print(row)

for oid, shape in fc1[('OID@', 'SHAPE@')]:
    print(f'{fc1.name} {oid} is {shape.length} {fc1.unit_name} long'

ref = SpatialReference(4326)
with fc1.projection_as(ref):
    for oid, shape in fc1[('OID@', 'SHAPE@')]:
        print(f'{fc1.name} {oid} is {shape.length} {ref.linearUnitName} long' 
```
