import folium
from folium import plugins
from branca import colormap as cm
from folium.plugins import MeasureControl, Fullscreen
import numpy as np
import pandas as pd

def create_detailed_solar_map(gdf):
    """
    Create interactive map with visualization of solar analysis results
    """
    print("Creating solar potential map...")

    try:
        # Data validation and cleaning for mapping
        gdf_clean = gdf[
            (gdf['solar_potential'] > 0) & 
            (gdf['effective_area'] > 0) & 
            (gdf['peak_power'] > 0) &
            (gdf['solar_potential'].notna())  # Exclude NaN values
        ].copy()

        if len(gdf_clean) == 0:
            raise ValueError("No valid buildings with solar potential found for mapping")

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

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            tiles='CartoDB dark_matter',
            prefer_canvas=True
        )

        # Fit bounds to show all data
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Calculate log-scaled solar potential
        gdf_wgs84['solar_potential_log'] = np.log1p(gdf_wgs84['solar_potential'])
        vmin_log = gdf_wgs84['solar_potential_log'].min()
        vmax_log = gdf_wgs84['solar_potential_log'].max()

        # Create color map with non-NaN tick labels
        tick_values = np.linspace(vmin_log, vmax_log, 7)
        tick_labels = [f"{np.exp(v):,.0f}" for v in tick_values if not np.isnan(v)]

        colormap = cm.LinearColormap(
            colors=['#1a1a1a', '#4d4d4d', '#808080', '#ffff99', '#ffcc00', '#ff9900', '#ff0000'],
            vmin=vmin_log,
            vmax=vmax_log,
            caption='Solar Potential (kWh/year)',
            tick_labels=tick_labels
        )

        # Add custom CSS
        m.get_root().html.add_child(folium.Element("""
            <style>
                .colormap-caption { color: white !important; }
                .tick text { fill: white !important; }
            </style>
        """))


        # Create feature groups
        buildings_fg = folium.FeatureGroup(name='Buildings')
        circles_fg = folium.FeatureGroup(name='Overview Markers')

        # Calculate circle scaling with smaller radii
        min_radius = 1
        max_radius = 8
        min_solar = gdf_wgs84['solar_potential'].min()
        max_solar = gdf_wgs84['solar_potential'].max()

        def scale_radius(value):
            return min_radius + (max_radius - min_radius) * (
                np.log1p(value) - np.log1p(min_solar)
            ) / (np.log1p(max_solar) - np.log1p(min_solar))

        # Add buildings to map
        for idx, row in gdf_wgs84.iterrows():
            try:
                color = colormap(row['solar_potential_log'])
                
                # Create tooltip with Name, Type, and Solar Potential
                tooltip_content = f"""
                Name: {row['name_right'] if 'name_right' in row else 'Unknown'}<br>
                Type: {row['fclass']}<br>
                Solar Potential: {row['solar_potential']:,.0f} kWh/year
                """

                popup_content = f"""
                <div style="width:300px; background-color:#1a1a1a; color:#ffffff; padding:10px; border-radius:5px">
                    <h4 style="color:#ffcc00">Solar Analysis Results</h4>
                    <b style="color:#ffcc00">Name:</b> {row['name_right'] if 'name_right' in row else 'Unknown'}<br>
                    <b style="color:#ffcc00">Type:</b> {row['fclass']}<br>
                    <b style="color:#ffcc00">Annual Generation:</b> {row['solar_potential']:,.0f} kWh/year<br>
                    <b style="color:#ffcc00">Peak Power:</b> {row['peak_power']:,.1f} kW<br>
                    <b style="color:#ffcc00">Shadow Factor:</b> {row['shadow_factor']:.2f}<br>
                    <hr style="border-color:#404040">
                    <b style="color:#ffcc00">Building Info:</b><br>
                    Height: {row['heightroof']} ft
                </div>
                """

                # Add building polygon
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': '#404040',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=tooltip_content
                ).add_to(buildings_fg)

                # Add circle marker
                centroid = row.geometry.centroid
                radius = scale_radius(row['solar_potential'])

                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=radius,
                    popup=popup_content,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=1,
                    tooltip=tooltip_content
                ).add_to(circles_fg)

            except Exception as e:
                print(f"Error adding building {idx} to map: {str(e)}")
                continue

        # Add layers and controls
        buildings_fg.add_to(m)
        circles_fg.add_to(m)
        colormap.add_to(m)

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

    except Exception as e:
        print(f"Error in map creation: {str(e)}")
        print("\nDetailed error information:")
        print(f"Number of buildings: {len(gdf)}")
        print(f"Buildings with positive solar potential: {(gdf['solar_potential'] > 0).sum()}")
        print(f"Buildings with positive effective area: {(gdf['effective_area'] > 0).sum()}")
        print(f"Buildings with positive peak power: {(gdf['peak_power'] > 0).sum()}")
        raise