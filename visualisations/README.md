# Visualisations

Folium map generation and other plotting scripts.

- `folium_trip_map.py` — renders GPS observations and the road network for a
  trip to an interactive HTML map.

```sh
python visualisations/folium_trip_map.py --trip "Student folder/train/00000001" \
  --output results/eda/sample_map.html
```

Static EDA previews already live in `results/eda/*.png`. An interactive Folium
HTML is still pending in the committed results.
