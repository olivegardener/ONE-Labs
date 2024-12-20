import folium
from folium import plugins
from branca import colormap as cm
from folium.plugins import MeasureControl, Fullscreen
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import rasterio
import shutil
import warnings
import os
import time

def create_detailed_heat_map(gdf, raster_path, raster_png=None):
    """
    Create an interactive map with visualization of heat analysis results.

    Parameters:
    - gdf: GeoDataFrame with 'heat_mean' and 'heat_index' columns
    - raster_path: Path to the original thermal raster (TIF)
    - raster_png: Optional path to a pre-generated PNG representation of the raster layer.
                  If provided, will be overlaid on the map.
    """

    print("Creating heat potential map...")

    # Data validation and cleaning for mapping
    gdf_clean = gdf[
        (gdf['heat_mean'].notna()) &
        (gdf['heat_index'].notna())
    ].copy()

    if len(gdf_clean) == 0:
        raise ValueError("No valid buildings with heat data found for mapping")

    # Convert to WGS84 for mapping
    gdf_wgs84 = gdf_clean.to_crs('EPSG:4326')

    # Calculate bounds for automatic zoom
    bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    if np.isnan(center_lat) or np.isnan(center_lon):
        center_lat = 40.7128
        center_lon = -74.0060
        print("Warning: Using default NYC coordinates for map center")

    # Create base map with dark background
    m = folium.Map(
        location=[center_lat, center_lon],
        tiles='CartoDB dark_matter',
        prefer_canvas=True
    )

    # Fit bounds to show all data
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Use 'heat_mean' attribute for coloring
    # Create a log or linear scale if desired. Here, we simply do a linear range.
    min_val = gdf_wgs84['heat_mean'].min()
    max_val = gdf_wgs84['heat_mean'].max()

    # Apply log if needed. Here, linear scale is okay.
    # Use the provided color ramp:
    # Dark Gray (#1a1a1a), Med-Dark Gray (#4d4d4d), Gray (#808080),
    # Pale Yellow (#ffff99), Yellow (#ffcc00), Orange (#ff9900), Red (#ff0000)
    value_breaks = np.quantile(gdf_wgs84['heat_mean'], [0, 0.16, 0.33, 0.5, 0.67, 0.84, 1])
    log_breaks = value_breaks  # no log transform needed, but we keep name for clarity

    colors = ['#1a1a1a', '#4d4d4d', '#808080', '#ffff99', '#ffcc00', '#ff9900', '#ff0000']

    # Create colormap
    colormap = cm.LinearColormap(
        colors=colors,
        index=log_breaks,
        vmin=log_breaks[0],
        vmax=log_breaks[-1],
        caption='Heat Mean (°F)'
    )

    # Add colormap to map
    colormap.add_to(m)

    # Optionally add raster as an overlay if PNG is provided
    # To create raster_png, you'd need to do separate processing:
    # For example, using rasterio to read a small overview and export as PNG.
    # This code assumes raster_png is a pre-made file aligned with EPSG:4326.
    if raster_png and Path(raster_png).exists():
        # Determine raster bounds (assuming it's already in EPSG:4326)
        with rasterio.open(raster_path) as src:
            # Transform not necessarily EPSG:4326. If not, you'd need reproject first.
            # Here we assume raster_png already matches WGS84 bounding box.
            # If not, reproject the raster before creating PNG.
            rb = src.bounds
        # Add raster overlay
        folium.raster_layers.ImageOverlay(
            name='Heat Raster',
            image=raster_png,
            bounds=[[rb.bottom, rb.left], [rb.top, rb.right]],
            opacity=0.7
        ).add_to(m)
    else:
        print("No raster PNG provided, skipping raster overlay.")

    # Create feature group for buildings
    buildings_fg = folium.FeatureGroup(name='Sites')

    # Add buildings to map
    for idx, row in gdf_wgs84.iterrows():
        try:
            # Map color based on heat_mean
            val = row['heat_mean']
            # Map value to color using colormap
            color = colormap(val)

            name = row['name'] if 'name' in row and pd.notnull(row['name']) else 'Unknown'
            fclass = row['fclass'] if 'fclass' in row and pd.notnull(row['fclass']) else 'Unknown'
            heat_mean = row['heat_mean']
            heat_index = row['heat_index']

            # Tooltip and popup content
            tooltip_content = f"""
            <b>Name:</b> {name}<br>
            <b>Type:</b> {fclass}<br>
            <b>Heat Mean:</b> {heat_mean:.1f} °F<br>
            <b>Heat Index:</b> {heat_index:.2f}
            """

            popup_content = f"""
            <div style="width:300px; background-color:#1a1a1a; color:#ffffff; padding:10px; border-radius:5px">
                <h4 style="color:#ffcc00">Heat Analysis Results</h4>
                <b style="color:#ffcc00">Name:</b> {name}<br>
                <b style="color:#ffcc00">Type:</b> {fclass}<br>
                <b style="color:#ffcc00">Heat Mean:</b> {heat_mean:.1f} °F<br>
                <b style="color:#ffcc00">Heat Index (Percentile):</b> {heat_index:.2f}<br>
                <hr style="border-color:#404040">
            </div>
            """

            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color: {
                    'fillColor': c,
                    'color': '#404040',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=tooltip_content
            ).add_to(buildings_fg)

        except Exception as e:
            print(f"Error adding building {idx} to map: {str(e)}")
            continue

    buildings_fg.add_to(m)

    # Add controls
    m.add_child(MeasureControl(
        position='topleft',
        primary_length_unit='miles',
        secondary_length_unit='feet',
        primary_area_unit='acres',
        secondary_area_unit='sqfeet'
    ))

    Fullscreen().add_to(m)
    folium.LayerControl(position='topright').add_to(m)

    return m

if __name__ == "__main__":
    # Load dataset
    input_file = Path('output') / 'sites_heat.geojson'
    if not input_file.exists():
        print("sites_heat.geojson not found. Please run heat_analysis.py first.")
        exit(1)

    # Load the heat-analyzed buildings
    analyzed_buildings = gpd.read_file(input_file)

    # Path to original raster
    raster_file = Path('input') / 'Landsat9_ThermalComposite_ST_B10_2020-2023.tif'

    # Optional: Path to a PNG overlay of the raster (if prepared)
    # If you have no PNG, set this to None. Otherwise, create a PNG from your raster externally.
    raster_png = None

    # Create and display the map
    heat_map = create_detailed_heat_map(analyzed_buildings, raster_file, raster_png=raster_png)

    # Save the map
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    map_file = output_dir / 'heat_potential_map.html'
    heat_map.save(str(map_file))

    print(f"\nInteractive heat map saved to {map_file}")