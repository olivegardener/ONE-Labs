import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.features import GeoJsonTooltip, GeoJsonPopup
from branca.colormap import linear
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px

# ---------------------------
# Configuration
# ---------------------------
invert_in_fields = True

# Default weights (must sum to 1.0)
default_weights = {
    "Adaptability_Index_weight": 0.25,
    "Solar_Energy_Index_weight": 0.15,
    "Heat_Hazard_Index_weight": 0.00,
    "Flood_Hazard_Index_weight": 0.10,
    "Heat_Vulnerability_Index_weight": 0.15,
    "Flood_Vulnerability_Index_weight": 0.15,
    "Service_Population_Index_weight": 0.20
}

sub_index_names = [
    "Adaptability_Index",
    "Solar_Energy_Index",
    "Heat_Hazard_Index",
    "Flood_Hazard_Index",
    "Heat_Vulnerability_Index",
    "Flood_Vulnerability_Index",
    "Service_Population_Index"
]

# Updated color scheme
index_colors = {
    "Adaptability_Index": "#FF6B6B",        # coral red
    "Solar_Energy_Index": "#4ECDC4",        # turquoise
    "Heat_Hazard_Index": "#FFD93D",         # bright yellow
    "Flood_Hazard_Index": "#6C5CE7",        # purple
    "Heat_Vulnerability_Index": "#A8E6CF",   # mint green
    "Flood_Vulnerability_Index": "#FF8B94",  # pink
    "Service_Population_Index": "#98DFAF"    # sage green
}

components = {
    "Adaptability_Index_components": {
        "RS_Priority": 0.25,
        "CAPACITY": 0.25,
        "BldgArea": 0.25,
        "StrgeArea": 0.25
    },
    "Solar_Energy_Index_components": {
        "peak_power": 1.0
    },
    "Heat_Hazard_Index_components": {
        "heat_mean": 1.0
    },
    "Flood_Hazard_Index_components": {
        "Cst_500_in": 0.1,
        "Cst_500_nr": 0.1,
        "Cst_100_in": 0.1,
        "Cst_100_nr": 0.1,
        "StrmShl_in": 0.1,
        "StrmShl_nr": 0.1,
        "StrmDp_in": 0.1,
        "StrmDp_nr": 0.1,
        "StrmTid_in": 0.1,
        "StrmTid_nr": 0.1
    },
    "Heat_Vulnerability_Index_components": {
        "hvi_area": 1.0
    },
    "Flood_Vulnerability_Index_components": {
        "ssvul_area": 0.5,
        "tivul_area": 0.5
    },
    "Service_Population_Index_components": {
        "pop_est": 1.0
    }
}


def min_max_normalize(series):
    s = series.fillna(0).astype(float)
    s_min = s.min()
    s_max = s.max()
    if s_min == s_max:
        return s.apply(lambda x: 0.0 if s_max == 0 else 1.0)
    return (s - s_min) / (s_max - s_min)


def create_sub_index(gdf, components_dict, invert_in=False):
    sub_index = np.zeros(len(gdf))
    for col, w in components_dict.items():
        if col not in gdf.columns:
            col_data = pd.Series(np.zeros(len(gdf)), index=gdf.index)
        else:
            col_data = pd.to_numeric(gdf[col], errors='coerce').fillna(0)
        norm_vals = min_max_normalize(col_data)
        if invert_in and col.endswith("_in"):
            norm_vals = 1 - norm_vals
        sub_index += (norm_vals * w)
    return sub_index

def calculate_indices(gdf, weights):
    Adaptability_Index = create_sub_index(gdf, components["Adaptability_Index_components"], invert_in=False)
    Solar_Energy_Index = create_sub_index(gdf, components["Solar_Energy_Index_components"], invert_in=False)
    Heat_Hazard_Index = create_sub_index(gdf, components["Heat_Hazard_Index_components"], invert_in=False)
    Flood_Hazard_Index = create_sub_index(gdf, components["Flood_Hazard_Index_components"], invert_in=invert_in_fields)
    Heat_Vulnerability_Index = create_sub_index(gdf, components["Heat_Vulnerability_Index_components"], invert_in=False)
    Flood_Vulnerability_Index = create_sub_index(gdf, components["Flood_Vulnerability_Index_components"], invert_in=False)
    Service_Population_Index = create_sub_index(gdf, components["Service_Population_Index_components"], invert_in=False)
    
    Suitability_Index = (
        Adaptability_Index * weights["Adaptability_Index_weight"] +
        Solar_Energy_Index * weights["Solar_Energy_Index_weight"] +
        Heat_Hazard_Index * weights["Heat_Hazard_Index_weight"] +
        Flood_Hazard_Index * weights["Flood_Hazard_Index_weight"] +
        Heat_Vulnerability_Index * weights["Heat_Vulnerability_Index_weight"] +
        Flood_Vulnerability_Index * weights["Flood_Vulnerability_Index_weight"] +
        Service_Population_Index * weights["Service_Population_Index_weight"]
    )

    index_norm = min_max_normalize(pd.Series(Suitability_Index))
    gdf["Adaptability_Index"] = Adaptability_Index
    gdf["Solar_Energy_Index"] = Solar_Energy_Index
    gdf["Heat_Hazard_Index"] = Heat_Hazard_Index
    gdf["Flood_Hazard_Index"] = Flood_Hazard_Index
    gdf["Heat_Vulnerability_Index"] = Heat_Vulnerability_Index
    gdf["Flood_Vulnerability_Index"] = Flood_Vulnerability_Index
    gdf["Service_Population_Index"] = Service_Population_Index
    gdf["Suitability_Index"] = Suitability_Index
    gdf["index_norm"] = index_norm
    return gdf

# Function to format fields (from the working map code)
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

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(layout="wide")

# Remove extra padding
st.markdown("""
<style>
.main > div {
    padding: 0;
    margin: 0;
}
.block-container {
    padding: 0;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>NYC Resilience Hub Prioritization Tool</h1>", unsafe_allow_html=True)

# Load both GeoJSON files
primary_path = "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/RH_Primary_Sites.geojson"
secondary_path = "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/RH_Secondary_Sites.geojson"

# Sidebar with sliders
st.sidebar.markdown("## Sub-Index Weights")

weights_input = {}
for idx_name in sub_index_names:
    default_val = default_weights[f"{idx_name}_weight"]
    val = st.sidebar.slider(
        idx_name.replace("_", " "), 
        min_value=0.0, max_value=1.0, value=default_val, step=0.05, key=idx_name
    )
    weights_input[f"{idx_name}_weight"] = val

recalc = st.sidebar.button("Recalculate")

# Use session_state to store calculated results
if "gdf_results" not in st.session_state or recalc:
    # Load and process data
    gdf_primary = gpd.read_file(primary_path)
    gdf_secondary = gpd.read_file(secondary_path)

    # Format fields for both datasets
    gdf_primary = format_fields(gdf_primary)
    gdf_secondary = format_fields(gdf_secondary)

    # Ensure WGS84 projection
    gdf_primary = gdf_primary.to_crs(epsg=4326)
    gdf_secondary = gdf_secondary.to_crs(epsg=4326)

    # Store in session state
    st.session_state["gdf_primary"] = gdf_primary
    st.session_state["gdf_secondary"] = gdf_secondary
    st.session_state["weights"] = normalized_weights
else:
    # Use existing results
    gdf_primary = st.session_state["gdf_primary"]
    gdf_secondary = st.session_state["gdf_secondary"]
    normalized_weights = st.session_state["weights"]

# Create layout
col1, col2 = st.columns([1,5])

# Donut chart of weights
weights_df = pd.DataFrame({
    "index": [n.replace("_", " ").replace(" Index", "") for n in sub_index_names],
    "weight": [normalized_weights[f"{n}_weight"] for n in sub_index_names],
    "color": [index_colors[n] for n in sub_index_names]
})

fig = px.pie(weights_df, 
             values='weight', 
             names='index', 
             hole=0.4,
             color='index',
             color_discrete_sequence=weights_df['color'].tolist())

fig.update_layout(
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        font=dict(color="white")
    ),
    paper_bgcolor="#222222",
    plot_bgcolor="#222222",
    font_color="white",
    margin=dict(t=20, b=80, l=20, r=20),
    width=None,
    height=400
)

with col1:
    st.plotly_chart(fig, use_container_width=True)
    
with col2:
    # Initialize map
    m = folium.Map(tiles='CartoDB dark_matter', zoom_start=4)
    colormap = linear.YlGn_09.scale(0, 1)
    colormap.position = 'topright'
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

    # Add primary sites layer
    primary_layer = folium.GeoJson(
        data=gdf_primary.__geo_interface__,
        name='Primary Sites',
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
                'pop_est'
            ],
            aliases=[
                'Address:', 
                'Building Area (sq ft):', 
                'Capacity:', 
                'Number of Stories:',
                'Year Built:',
                'Owner:',
                'Operation Type:',
                'Population within 2000ft:'
            ],
            sticky=True,
            labels=True,
        )
    ).add_to(m)

    # Add secondary sites layer
    secondary_layer = folium.GeoJson(
        data=gdf_secondary.__geo_interface__,
        name='Secondary Sites',
        style_function=style_function,
        tooltip=tooltip,
        popup=popup
    ).add_to(m)

    # Add colormap and layer control
    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    # Calculate bounds from both datasets
    bounds = gpd.GeoDataFrame(pd.concat([gdf_primary, gdf_secondary])).total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Display the map
    st_map = st_folium(m, width=900, height=700)