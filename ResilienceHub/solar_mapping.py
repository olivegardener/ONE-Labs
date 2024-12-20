import folium
from folium import plugins
from branca import colormap as cm
from folium.plugins import MeasureControl, Fullscreen
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
import shutil

def create_detailed_solar_map(buildings_gdf, points_gdf):
    """
    Create interactive map with visualization of solar analysis results.
    Only points will have tooltips and popups, and they will include name and OwnerName.
    The legend will correctly show the range of solar potential.
    """
    print("Creating solar potential map...")
    try:
        # Data validation and cleaning for mapping
        buildings_clean = buildings_gdf[
            (buildings_gdf['solar_potential'] > 0) &
            (buildings_gdf['effective_area'] > 0) &
            (buildings_gdf['peak_power'] > 0) &
            (buildings_gdf['solar_potential'].notna())
        ].copy()

        if len(buildings_clean) == 0 and len(points_gdf) == 0:
            raise ValueError("No valid buildings or points with solar potential found for mapping")

        # Convert to WGS84 for mapping
        if not buildings_clean.empty:
            buildings_wgs84 = buildings_clean.to_crs('EPSG:4326')
        else:
            # If no buildings, create an empty geodataframe
            buildings_wgs84 = gpd.GeoDataFrame(crs='EPSG:4326', geometry=[])

        points_wgs84 = points_gdf.to_crs('EPSG:4326')

        # Calculate bounds for automatic zoom
        # Combine bounds of buildings and points
        all_bounds = []
        if not buildings_wgs84.empty:
            all_bounds.append(buildings_wgs84.total_bounds)
        if not points_wgs84.empty:
            all_bounds.append(points_wgs84.total_bounds)

        if all_bounds:
            combined_bounds = np.vstack(all_bounds)
            minx = np.min(combined_bounds[:,0])
            miny = np.min(combined_bounds[:,1])
            maxx = np.max(combined_bounds[:,2])
            maxy = np.max(combined_bounds[:,3])
            bounds = [minx, miny, maxx, maxy]
        else:
            # Default to NYC if no data
            bounds = [-74.0060, 40.7128, -74.0060, 40.7128]

        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2

        if np.isnan(center_lat) or np.isnan(center_lon):
            center_lat = 40.7128
            center_lon = -74.0060
            print("Warning: Using default NYC coordinates for map center")

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            tiles='CartoDB Positron',
            prefer_canvas=True,
            zoom_start=13
        )

        # Fit bounds to show all data if available
        if not np.isnan(center_lat) and not np.isnan(center_lon):
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Gather all valid solar potential values (from buildings and points) for colormap
        building_values = buildings_wgs84['solar_potential'].dropna().values if not buildings_wgs84.empty else np.array([])
        point_values = points_wgs84['solar_potential'].dropna().values if not points_wgs84.empty else np.array([])

        valid_values = np.concatenate([building_values, point_values])
        valid_values = valid_values[valid_values > 0]

        if len(valid_values) == 0:
            raise ValueError("No valid solar potential values found for color mapping")

        # Compute log scale for colormap
        min_val = valid_values.min()
        max_val = valid_values.max()
        log_min = np.log1p(min_val)
        log_max = np.log1p(max_val)

        # We have 7 colors, so we create 7 breaks from min to max in log space
        colors = ['#4d4d4d', '#fff7bc', '#fee391', '#fdb863', '#f87d43', '#e95224', '#cc3311']
        log_breaks = np.linspace(log_min, log_max, len(colors))
        value_breaks = np.expm1(log_breaks)

        # Create tick labels from value_breaks
        tick_labels = [
            f"{value/1000000:.1f}M" if value >= 1000000 else
            f"{value/1000:.0f}K" if value >= 1000 else
            f"{value:.0f}"
            for value in value_breaks
        ]

        # Create color map with specified colors and breaks
        colormap = cm.LinearColormap(
            colors=colors,
            index=log_breaks,
            vmin=log_breaks[0],
            vmax=log_breaks[-1],
            caption='Solar Potential (kWh/year)',
            tick_labels=tick_labels
        )
        colormap.add_to(m)

        # Create feature groups
        buildings_fg = folium.FeatureGroup(name='Buildings', show=True)
        points_fg = folium.FeatureGroup(name='Points', show=True)

        def map_color(value):
            """Convert solar potential value to color using log-adjusted colormap"""
            if pd.isna(value) or value <= 0:
                return '#4d4d4d'
            log_value = np.log1p(value)
            return colormap(log_value)

        def calculate_radius(value):
            """Calculate circle radius based on log-adjusted solar potential"""
            if pd.isna(value) or value <= 0:
                return 2

            # Log-scale the value
            log_value = np.log1p(value)
            log_min_val = np.log1p(points_wgs84['solar_potential'].min())
            log_max_val = np.log1p(points_wgs84['solar_potential'].max())

            # Use exponential scaling for more dramatic size differences
            min_radius = 1
            max_radius = 6
            normalized = (log_value - log_min_val) / (log_max_val - log_min_val)
            radius = min_radius + (np.exp(normalized * 2) - 1) * (max_radius - min_radius) / (np.e - 1)

            return radius

        # Add buildings to the map without popup or tooltip
        # Just a simple style
        if not buildings_wgs84.empty:
            folium.GeoJson(
                buildings_wgs84,
                style_function=lambda feature: {
                    'fillColor': map_color(feature['properties']['solar_potential']),
                    'color': '#666666',
                    'weight': 1,
                    'fillOpacity': 0.7
                }
            ).add_to(buildings_fg)

        # Add points to map with popup and tooltip
        for idx, point in points_wgs84.iterrows():
            try:
                color = map_color(point['solar_potential'])
                radius = calculate_radius(point['solar_potential'])

                name = point['name'] if pd.notna(point['name']) else 'N/A'
                type = point['fclass'] if pd.notna(point['fclass']) else 'N/A'
                owner = point['OwnerName'] if pd.notna(point['OwnerName']) else 'N/A'
                area = point['area_ft2'] if pd.notna(point['area_ft2']) else 'N/A'

                popup_content = f"""
                <div style="width:300px">
                    <h4>Type: {type}</h4>
                    <b>Name:</b> {name}<br>
                    <b>Owner:</b> {owner}<br>
                    <b>Area:</b> {area} sq ft<br>
                    <b>Annual Generation:</b> {point['solar_potential']:,.0f} kWh/year<br>
                    <b>Peak Power:</b> {point['peak_power']:,.1f} kW<br>
                </div>
                """

                tooltip = f"{type}: {point['solar_potential']:,.0f} kWh/year"

                folium.CircleMarker(
                    location=[point.geometry.y, point.geometry.x],
                    radius=radius,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.9,
                    weight=1,
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=tooltip
                ).add_to(points_fg)

            except Exception as e:
                print(f"Error adding point {idx} to map: {str(e)}")
                continue

        # Add the feature groups to the map
        buildings_fg.add_to(m)
        points_fg.add_to(m)

        # Add controls
        m.add_child(MeasureControl(
            position='topleft',
            primary_length_unit='miles',
            secondary_length_unit='feet',
            primary_area_unit='acres',
            secondary_area_unit='sqfeet'
        ))

        # Add layer control
        folium.LayerControl(position='topright').add_to(m)

        return m

    except Exception as e:
        print(f"Error in map creation: {str(e)}")
        raise

if __name__ == "__main__":
    # Load datasets
    buildings_file = Path('output') / 'sites_solar.geojson'
    points_file = Path('output') / 'sites_solar_points.geojson'

    if not buildings_file.exists():
        print("sites_solar.geojson not found. Please run solar_analyzer.py first.")
        exit(1)
    if not points_file.exists():
        print("sites_solar_points.geojson not found. Please run solar_analyzer.py first.")
        exit(1)

    analyzed_buildings = gpd.read_file(buildings_file)
    analyzed_points = gpd.read_file(points_file)

    # Create the deployment directory
    solarwebmap_deploy_dir = Path('output') / 'solarwebmap-deploy'
    solarwebmap_deploy_dir.mkdir(parents=True, exist_ok=True)

    # Create and save the map
    solar_map = create_detailed_solar_map(analyzed_buildings, analyzed_points)
    map_file = solarwebmap_deploy_dir / 'solar_potential_map.html'
    solar_map.save(str(map_file))
    print(f"\nInteractive map saved to {map_file}")

    # Copy as index.html
    index_file = solarwebmap_deploy_dir / 'index.html'
    shutil.copy(str(map_file), str(index_file))
    print(f"Webmap deployed at {index_file}")