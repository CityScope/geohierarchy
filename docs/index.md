# geohierarchy

Keep the same data at several geographic levels (counties, tracts, block groups, H3 cells, streets...) and fill every level
automatically.

You add each level once, say which levels contain which, and say how each column moves between levels:

* **up** (many small units into one big unit): add counts with `Sum`, average rates with `Mean`, or take the `Max` / `Min`;
* **down** (one big unit into many small ones): split a count in proportion to a weight (area, population...) or copy an
  average.

geohierarchy then fills every column at every level it is missing from. A level's own data are never overwritten.

## Install

```bash
pip install -e .        # from the repository folder; Python 3.10 or newer
```

Maps (`geohierarchy.maps`) need the `folium` / `branca` extras; plain `import geohierarchy` does not load them.

## Example

```python
from geohierarchy import GeoHierarchy, Sum, Mean

gh = GeoHierarchy(crs="EPSG:4326")
gh.add_level("county", county_gdf, id_col="GEOID", agg=Sum())
gh.add_level("tract", tract_gdf, id_col="GEOID", agg=Sum(), parent="county")
gh.add_level("block", block_gdf, id_col="GEOID", agg=Sum(), parent="tract")

# data from outside the hierarchy, averaged onto tracts weighted by population
gh.add_vector_data(streets_gdf, level="tract", agg=Mean(weight_column="population_total"))

gh["tract"]   # every column, filled in
```

## Where to go next

* [Guide](guide.md): the concepts (levels, the graph, strategies, H3 and raster data) with a worked example.
* [API reference](api/index.md): every public class and function.

## Caching

Nothing is written to disk unless you ask. The id-mapping cache (`geohierarchy.spatial_cache`) takes an explicit
`cache_dir`; pass a folder you keep to reuse mappings between runs.
