# ---------------------------
# Imports
# ---------------------------
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.features import GeoJsonTooltip, GeoJsonPopup
from branca.colormap import linear
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import os

# ---------------------------
# Streamlit Configuration
# ---------------------------
st.set_page_config(layout="wide", page_title="NYC Resilience Hub Prioritization Tool")

# ---------------------------
# Application Constants
# ---------------------------
# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "primary": os.path.join(SCRIPT_DIR, "output", "RH_Primary_Sites.geojson"),
    "secondary": os.path.join(SCRIPT_DIR, "output", "RH_Secondary_Sites.geojson")
}

# Index configuration
INDEX_CONFIG = {
    "weights": {
        "Adaptability_Index_weight": 0.25,
        "Solar_Energy_Index_weight": 0.15,
        "Heat_Hazard_Index_weight": 0.00,
        "Flood_Hazard_Index_weight": 0.10,
        "Heat_Vulnerability_Index_weight": 0.15,
        "Flood_Vulnerability_Index_weight": 0.15,
        "Service_Population_Index_weight": 0.20
    },
    "names": [
        "Adaptability_Index",
        "Solar_Energy_Index",
        "Heat_Hazard_Index",
        "Flood_Hazard_Index",
        "Heat_Vulnerability_Index",
        "Flood_Vulnerability_Index",
        "Service_Population_Index"
    ],
    "colors": {
        "Adaptability_Index": "#FF6B6B",
        "Solar_Energy_Index": "#4ECDC4",
        "Heat_Hazard_Index": "#FFD93D",
        "Flood_Hazard_Index": "#6C5CE7",
        "Heat_Vulnerability_Index": "#A8E6CF",
        "Flood_Vulnerability_Index": "#FF8B94",
        "Service_Population_Index": "#98DFAF"
    },
    "labels": {
        "Adaptability_Index": "Building Adaptability",
        "Solar_Energy_Index": "Solar Potential",
        "Heat_Hazard_Index": "Heat Risk",
        "Flood_Hazard_Index": "Flood Risk & Proximity",
        "Heat_Vulnerability_Index": "Heat Vulnerability",
        "Flood_Vulnerability_Index": "Flood Vulnerability",
        "Service_Population_Index": "Population within 2000ft"
    },
    "descriptions": {
        "Adaptability_Index": "Building characteristics that make it suitable for adaptation",
        "Solar_Energy_Index": "Potential for solar power generation",
        "Heat_Hazard_Index": "Level of heat risk in the area",
        "Flood_Hazard_Index": "Combines flood risk (negative) and proximity to flood zones (positive)",
        "Heat_Vulnerability_Index": "Social vulnerability to heat impacts",
        "Flood_Vulnerability_Index": "Social vulnerability to flood impacts",
        "Service_Population_Index": "Population within service area"
    }
}

COMPONENT_LABELS = {
    "Adaptability_Index_components": {
        "RS_Priority": "Building Type Prioritization",
        "CAPACITY": "Building Capacity (number of people)",
        "BldgArea": "Building Area",
        "StrgeArea": "Storage Area"
    },
    "Solar_Energy_Index_components": {
        "peak_power": "Peak Solar Power Potential"
    },
    "Heat_Hazard_Index_components": {
        "heat_mean": "Heat Risk Score"
    },
    "Flood_Hazard_Index_components": {
        "Cst_500_in": "Flooding in Building (500-year flood)",
        "Cst_500_nr": "Flooding near Building (500-year flood)",
        "Cst_100_in": "Flooding in Building (100-year flood)",
        "Cst_100_nr": "Flooding near Building (100-year flood)",
        "StrmShl_in": "Shallow Flooding in Building (Stormwater 2080)",
        "StrmShl_nr": "Shallow Flooding near Building (Stormwater 2080)",
        "StrmDp_in": "Deep Flooding in Building (Stormwater 2080)",
        "StrmDp_nr": "Deep Flooding near Building (Stormwater 2080)",
        "StrmTid_in": "Flooding in Building (High Tide 2080)",
        "StrmTid_nr": "Flooding near Building (High Tide 2080)"
    },
    "Heat_Vulnerability_Index_components": {
        "hvi_area": "Heat Vulnerability Score"
    },
    "Flood_Vulnerability_Index_components": {
        "ssvul_area": "Nearby Storm Surge Flooding Vulnerability",
        "tivul_area": "Nearby Tidal Flooding Vulnerability"
    },
    "Service_Population_Index_components": {
        "pop_est": "Population Count"
    }
}

# Component weights configuration
COMPONENTS = {
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

# ---------------------------
# Utility Functions
# ---------------------------
def min_max_normalize(series):
    """Normalize a pandas series to range [0,1]"""
    s = series.fillna(0).astype(float)
    s_min = s.min()
    s_max = s.max()
    if s_min == s_max:
        return s.apply(lambda x: 0.0 if s_max == 0 else 1.0)
    return (s - s_min) / (s_max - s_min)

def create_sub_index(gdf, components_dict, flood_hazard=False):
    """
    Create a sub-index based on multiple fields.

    Args:
        gdf: GeoDataFrame containing the component fields
        components_dict: Dictionary of component names and their weights
        flood_hazard: Boolean indicating if this is a flood hazard index
    """
    sub_index = np.zeros(len(gdf))

    for col, w in components_dict.items():
        if col not in gdf.columns:
            col_data = pd.Series(np.zeros(len(gdf)), index=gdf.index)
        else:
            col_data = pd.to_numeric(gdf[col], errors='coerce').fillna(0)

        norm_vals = min_max_normalize(col_data)

        if flood_hazard and col.endswith("_in"):
            norm_vals = 1 - norm_vals

        sub_index += (norm_vals * w)
    return sub_index

def calculate_indices(gdf, weights):
    """Calculate all indices for a given GeoDataFrame"""
    indices = {
        "Adaptability_Index": create_sub_index(gdf, COMPONENTS["Adaptability_Index_components"]),
        "Solar_Energy_Index": create_sub_index(gdf, COMPONENTS["Solar_Energy_Index_components"]),
        "Heat_Hazard_Index": create_sub_index(gdf, COMPONENTS["Heat_Hazard_Index_components"]),
        "Flood_Hazard_Index": create_sub_index(gdf, COMPONENTS["Flood_Hazard_Index_components"], flood_hazard=True),
        "Heat_Vulnerability_Index": create_sub_index(gdf, COMPONENTS["Heat_Vulnerability_Index_components"]),
        "Flood_Vulnerability_Index": create_sub_index(gdf, COMPONENTS["Flood_Vulnerability_Index_components"]),
        "Service_Population_Index": create_sub_index(gdf, COMPONENTS["Service_Population_Index_components"])
    }

    # Calculate Suitability Index
    Suitability_Index = sum(index * weights[f"{name}_weight"] for name, index in indices.items())

    # Add all indices to GeoDataFrame
    for name, index in indices.items():
        gdf[name] = index

    gdf["Suitability_Index"] = Suitability_Index
    gdf["index_norm"] = min_max_normalize(pd.Series(Suitability_Index))

    return gdf

def format_fields(gdf):
    """Format fields for display"""
    # Fields to convert zeros to 'Unknown'
    zero_to_unknown_fields = ['CAPACITY', 'BldgArea', 'num_story', 'cnstrct_yr']
    for field in zero_to_unknown_fields:
        if field in gdf.columns:
            gdf[field] = gdf[field].apply(lambda x: 'Unknown' if pd.isna(x) or x == 0 else x)

    # Format index fields to 2 decimal places
    index_fields = ['Adaptability_Index', 'Solar_Energy_Index', 'Heat_Vulnerability_Index', 'Flood_Vulnerability_Index']
    for field in index_fields:
        if field in gdf.columns:
            gdf[field] = gdf[field].apply(lambda x: f"{float(x):.2f}" if pd.notnull(x) else 'Unknown')

    # Format population estimates with commas
    if 'pop_est' in gdf.columns:
        gdf['pop_est'] = gdf['pop_est'].apply(lambda x: f"{int(str(x).replace(',', '')):,}" if pd.notnull(x) and x != 0 else 'Unknown')

    # Format text fields
    for field in ['OwnerName', 'OPTYPE']:
        if field in gdf.columns:
            gdf[field] = gdf[field].apply(lambda x: 'Unknown' if pd.isna(x) or x == 'Unknown' else x)

    return gdf

def style_function(feature):
    """Style function for GeoJSON features"""
    try:
        value = float(feature['properties']['index_norm'])
        color = colormap(value)  # Get color from colormap
        return {
            'fillColor': color,
            'color': color,      # Apply same color to outline
            'weight': 2,         # Increase outline weight for better visibility
            'fillOpacity': 0.5,
            'opacity': 1         # Make outline fully opaque
        }
    except:
        return {
            'fillColor': '#808080',
            'color': '#808080',  # Match outline to fill for error cases
            'weight': 2,
            'fillOpacity': 0.5,
            'opacity': 1
        }


def add_site_layer(gdf, name, m):
    """Add a GeoJSON layer to the map"""
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
                'Address', 'BldgArea', 'CAPACITY', 'num_story', 'cnstrct_yr',
                'OwnerName', 'OPTYPE', 'pop_est', 'Adaptability_Index',
                'Solar_Energy_Index', 'Heat_Hazard_Index', 'Flood_Hazard_Index',
                'Heat_Vulnerability_Index', 'Flood_Vulnerability_Index',
                'Service_Population_Index'
            ],
            aliases=[
                'Address:', 'Building Area (sq ft):', 'Capacity:', 
                'Number of Stories:', 'Year Built:', 'Owner:', 'Operation Type:',
                'Population within 2000ft:', 'Adaptability Index:',
                'Solar Potential Index:', 'Heat Risk Index:',
                'Flood Risk & Proximity Index:', 'Heat Vulnerability Index:',
                'Flood Vulnerability Index:', 'Population Served Index:'
            ],
            sticky=True,
            labels=True,
        )
    ).add_to(m)

# ---------------------------
# CSS Styling
# ---------------------------
CUSTOM_CSS = """
.main { padding: 0rem; }
.block-container {
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 0rem;
}
.element-container { padding: 0.5rem; }
div[data-testid="stSidebarNav"] { padding-top: 0rem; }
div[data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }
div[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding-top: 0rem;
    padding-bottom: 0rem;
}
div[data-testid="column"] {
    padding: 0rem;
    margin: 0rem;
}
iframe {
    width: 100%;
    min-height: 800px;
    height: 100%;
    border: none;
}
.stSlider {
    padding-left: 1rem;
    padding-right: 1rem;
}
.css-1544g2n { padding-top: 0rem; }
.css-1544g2n > div { padding-top: 0rem; }
h1 {
    padding-top: 0rem;
    padding-bottom: 1rem;
}
h3 { padding-bottom: 0.5rem; }
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ---------------------------
# Main Application Layout
# ---------------------------
st.markdown("<h1 style='text-align: left; color: white;'>NYC Resilience Hub Prioritization Tool</h1>", unsafe_allow_html=True)

# ---------------------------
# Sidebar Configuration
# ---------------------------
st.sidebar.markdown("## Index Weights")
st.sidebar.markdown("Adjust the weights below to customize the importance of each factor.")

# Initialize dictionary to store weight inputs
weights_input = {}

# Function to create expandable section for each index
def create_index_section(index_name, friendly_name):
    """
    Creates an expandable section in the sidebar for each index and its sub-components

    Args:
        index_name: The technical name of the index (e.g., "Adaptability_Index")
        friendly_name: The display name of the index (e.g., "Building Adaptability")
    """
    with st.sidebar.expander(friendly_name, expanded=False):
        # Create the main index weight slider
        val = st.slider(
            "Overall Weight",
            min_value=0.0,
            max_value=1.0,
            value=INDEX_CONFIG["weights"][f"{index_name}_weight"],
            step=0.05,
            help=INDEX_CONFIG["descriptions"][index_name],
            key=index_name
        )
        weights_input[f"{index_name}_weight"] = val

        # If this index has sub-components, create sliders for each
        if f"{index_name}_components" in COMPONENTS:
            st.markdown("#### Sub-components")
            for comp, default_weight in COMPONENTS[f"{index_name}_components"].items():
                friendly_comp_name = COMPONENT_LABELS[f"{index_name}_components"][comp]
                comp_val = st.slider(
                    friendly_comp_name,
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
                    step=0.05,
                    key=f"{index_name.lower()}_{comp}"
                )
                COMPONENTS[f"{index_name}_components"][comp] = comp_val

# Create sections for all indices
for idx_name in INDEX_CONFIG["names"]:
    create_index_section(idx_name, INDEX_CONFIG["labels"][idx_name])

# ---------------------------
# Data Processing
# ---------------------------
# Normalize weights
total_w = sum(weights_input.values())
normalized_weights = {k: v/total_w for k, v in weights_input.items()} if total_w > 0 else {k: 1.0/len(weights_input) for k in weights_input}

# Normalize component weights
for index_name, comp_dict in COMPONENTS.items():
    total = sum(comp_dict.values())
    if total > 0:
        COMPONENTS[index_name] = {k: v/total for k, v in comp_dict.items()}

# Load and process data
if "gdf_primary" not in st.session_state:
    gdf_primary = gpd.read_file(PATHS["primary"])
    gdf_secondary = gpd.read_file(PATHS["secondary"])

    # Ensure data is in EPSG:4326
    for gdf in [gdf_primary, gdf_secondary]:
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

    # Calculate indices and format fields
    for gdf in [gdf_primary, gdf_secondary]:
        gdf = calculate_indices(gdf, normalized_weights)
        gdf = format_fields(gdf)

    st.session_state.update({
        "gdf_primary": gdf_primary,
        "gdf_secondary": gdf_secondary,
        "weights": normalized_weights
    })
else:
    gdf_primary = st.session_state["gdf_primary"]
    gdf_secondary = st.session_state["gdf_secondary"]

    if normalized_weights != st.session_state["weights"]:
        for gdf in [gdf_primary, gdf_secondary]:
            gdf = calculate_indices(gdf, normalized_weights)
            gdf = format_fields(gdf)

        st.session_state.update({
            "gdf_primary": gdf_primary,
            "gdf_secondary": gdf_secondary,
            "weights": normalized_weights
        })

# ---------------------------
# Visualization
# ---------------------------
col1, col2 = st.columns([2, 5])

# Weights Distribution Pie Chart
with col1:
    st.markdown("### Weights Distribution")
    weights_df = pd.DataFrame({
        "index": [INDEX_CONFIG["labels"][n] for n in INDEX_CONFIG["names"]],
        "weight": [normalized_weights[f"{n}_weight"] for n in INDEX_CONFIG["names"]],
        "color": [INDEX_CONFIG["colors"][n] for n in INDEX_CONFIG["names"]]
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
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color="white", size=10)
        ),
        paper_bgcolor="#222222",
        plot_bgcolor="#222222",
        font_color="white",
        margin=dict(t=20, b=50, l=20, r=20),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# Map
with col2:
    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles='CartoDB positron'
    )

    colormap = linear.YlGn_09.scale(0, 1)
    colormap.caption = 'Suitability Index'

    add_site_layer(gdf_primary, 'Primary Sites', m)
    add_site_layer(gdf_secondary, 'Secondary Sites', m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    combined_gdf = gpd.GeoDataFrame(pd.concat([gdf_primary, gdf_secondary], ignore_index=True), crs=gdf_primary.crs)
    if not combined_gdf.empty:
        bounds = combined_gdf.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    st_map = st_folium(
        m,
        width='100%',
        height=800,
        returned_objects=[],
        use_container_width=True,
        key="main_map"
    )
