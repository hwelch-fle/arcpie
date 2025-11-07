# Basics
arcpie is designed to be an easy drop in alongside arcpy proper that aligns interaction with geospatial objects closer to the 
Python model.

Here's a simple example of summing up the total length of road in a city:
```python
from arcpy import Polyline, Polygon
from arcpie import FeatureClass

roads = FeatureClass[Polyline]('Roads')
cities = FeatureClass[Polygon]('Cities')

for city_name, city_boundary in cities[('Name', 'SHAPE@')]:
    with roads.spatial_filter(city_boundary): # Apply a spatial filter
        road_length = sum(
            road.intersect(city, 2).length # Clip road to city limits
            for road in roads.shapes # Iterate the road shapes
        )
    print(f'{city_name} has {road_length} {roads.units} of road')
```

Here we accomplish quite a bit with a single simple comprehension and a flat feature iterator! Here's that same code written using base arcpy:

```python
from arcpy.da import SearchCursor

with SearchCursor('Cities', ['Name', 'SHAPE@']) as city_cur:
    for city_name, city_boundary in city_cur: # Iterate cities
        road_length = 0.0 # Initialize road length
        road_units = None
        with SearchCursor('Roads', ['SHAPE@'], spatial_filter=city_boundary) as road_cur: # Iterate Roads
            for road, in road_cur:
                if road_units is None:
                    road_units = road.spatialReference.linearUnitName
                road_length += road.intersect(city_boundary, 2).length # Clip and add

        print(f'{city_name} has {road_length} {road_units} of road')
```

Lets see what else we can do with that `FeatureClass` object:

```python
from arcpie import FeatureClass, RowRecord, FilterFunc, count

# Create a dynamic filter function that counts the number of
# lines in a text field
def line_count(field: str, line_count: int):
    def _inner(func: FilterFunc) -> bool:
        return len(row[field].splitlines()) == lines
    return _inner

notes = FeatureClass('FieldNotes')

print(count(notes[line_count('COMMENTS', 5)]))
# prints 50

print(count(notes[line_count('COMMENTS', 2)]))
# prints 245
```

Here we used a function index that allows you to define a condition for row iteration. The `arcpie.count()` function is just an alias for `sum(1 for _ in Iterator)` since all FeatureClass indexes are iterators and do not have a known length until they are exhausted.