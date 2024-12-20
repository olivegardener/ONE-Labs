import folium
import geopandas as gpd
from branca.colormap import linear
import pandas as pd
import json

# Load both GeoJSON files
primary_path = "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/RH_Primary_Sites.geojson"
secondary_path = "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/RH_Secondary_Sites.geojson"

# Read both datasets
gdf_primary = gpd.read_file(primary_path)
gdf_secondary = gpd.read_file(secondary_path)

# Function to format fields
def format_fields(gdf):
    # Convert 0 to 'Unknown' for specific fields
    zero_to_unknown_fields = ['CAPACITY', 'BldgArea', 'num_story', 'cnstrct_yr']
    for field in zero_to_unknown_fields:
        gdf[field] = gdf[field].apply(lambda x: 'Unknown' if pd.isna(x) or x == 0 else x)

    # Format index scores to 2 decimal places
    index_fields = ['Adaptability_Index', 'Solar_Energy_Index', 'Heat_Vulnerability_Index', 'Flood_Vulnerability_Index']
    for field in index_fields:
        gdf[field] = gdf[field].apply(lambda x: f"{float(x):.2f}" if pd.notnull(x) else 'Unknown')

    # Format population estimate with commas
    gdf['pop_est'] = gdf['pop_est'].apply(lambda x: f"{int(x):,}" if pd.notnull(x) and x != 0 else 'Unknown')

    # Clean up owner and operation type fields
    gdf['OwnerName'] = gdf['OwnerName'].apply(lambda x: 'Unknown' if pd.isna(x) or x == 'Unknown' else x)
    gdf['OPTYPE'] = gdf['OPTYPE'].apply(lambda x: 'Unknown' if pd.isna(x) or x == 'Unknown' else x)

    return gdf

# Apply formatting to both datasets
gdf_primary = format_fields(gdf_primary)
gdf_secondary = format_fields(gdf_secondary)

# Ensure both are in WGS84
gdf_primary = gdf_primary.to_crs(epsg=4326)
gdf_secondary = gdf_secondary.to_crs(epsg=4326)

# Initialize map with NYC coordinates
m = folium.Map(
    location=[40.7128, -74.0060],
    zoom_start=11,
    tiles='CartoDB positron'
)

# Create colormap
colormap = linear.YlGn_09.scale(0, 1)
colormap.caption = 'Suitability Index'

def style_function(feature):
    try:
        value = float(feature['properties']['index_norm'])
        return {
            'fillColor': colormap(value),
            'color': '#000000',
            'weight': 1,
            'fillOpacity': 0.5
        }
    except:
        return {
            'fillColor': '#gray',
            'color': '#000000',
            'weight': 1,
            'fillOpacity': 0.5
        }

def add_site_layer(gdf, name):
    return folium.GeoJson(
        data=gdf.__geo_interface__,
        name=name,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['name', 'FACTYPE', 'index_norm'],
            aliases=['Name:', 'Facility Type:', 'Suitability Score:'],
            sticky=True
        ),
        popup=folium.GeoJsonPopup(
            fields=[
                'Address', 
                'BldgArea', 
                'CAPACITY', 
                'num_story',
                'cnstrct_yr',
                'OwnerName',
                'OPTYPE',
                'pop_est',
                'Adaptability_Index',
                'Solar_Energy_Index',
                'Heat_Vulnerability_Index',
                'Flood_Vulnerability_Index'
            ],
            aliases=[
                'Address:', 
                'Building Area (sq ft):', 
                'Capacity:', 
                'Number of Stories:',
                'Year Built:',
                'Owner:',
                'Operation Type:',
                'Population within 2000ft:',
                'Adaptability Score:',
                'Solar Potential Score:',
                'Heat Vulnerability Score:',
                'Flood Vulnerability Score:'
            ],
            sticky=True,
            labels=True,
        )
    ).add_to(m)

# Add layers
try:
    primary_layer = add_site_layer(gdf_primary, 'Primary Sites')
    print("Primary layer added successfully")
except Exception as e:
    print(f"Error adding primary layer: {str(e)}")

try:
    secondary_layer = add_site_layer(gdf_secondary, 'Secondary Sites')
    print("Secondary layer added successfully")
except Exception as e:
    print(f"Error adding secondary layer: {str(e)}")

# Add the colormap legend
colormap.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save the map
output_map = "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/RH_Combined_map.html"
m.save(output_map)

print("\nMap saved successfully")
