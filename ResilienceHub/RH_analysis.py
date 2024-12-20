import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime
import os
import shutil

# ---------------------------
# Configuration Dictionary
# ---------------------------
weights = {
    # Final index weights (must sum to 1.0)
    "Adaptability_Index_weight": 0.25,
    "Solar_Energy_Index_weight": 0.15,
    "Heat_Hazard_Index_weight": 0.00,
    "Flood_Hazard_Index_weight": 0.10,
    "Heat_Vulnerability_Index_weight": 0.15,
    "Flood_Vulnerability_Index_weight": 0.15,
    "Service_Population_Index_weight": 0.20,

    # Sub-index weighting:
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
    """
    Min-Max normalize a pandas Series to [0,1].
    Treat NA as 0 before normalization.
    """
    s = series.fillna(0).astype(float)
    s_min = s.min()
    s_max = s.max()
    if s_min == s_max:
        # If constant column, all zero => 0, else => all ones
        return s.apply(lambda x: 0.0 if s_max == 0 else 1.0)
    return (s - s_min) / (s_max - s_min)

def create_sub_index(gdf, components_dict, flood_hazard=False):
    """
    Create a sub-index based on multiple fields.
    Each field is normalized, then multiplied by its sub-weight.
    For flood hazard:
        - '_in' fields are inverted (1 - normalized_value) as they reduce suitability
        - '_nr' fields are not inverted as they increase suitability
    """
    sub_index = np.zeros(len(gdf))

    for col, w in components_dict.items():
        # If the column doesn't exist, use zeros
        if col not in gdf.columns:
            col_data = pd.Series(np.zeros(len(gdf)), index=gdf.index)
        else:
            # Convert to numeric locally, don't change the original gdf
            col_data = pd.to_numeric(gdf[col], errors='coerce').fillna(0)

        norm_vals = min_max_normalize(col_data)

        # For flood hazard, invert only the '_in' fields
        if flood_hazard and col.endswith("_in"):
            norm_vals = 1 - norm_vals

        sub_index += (norm_vals * w)
    return sub_index

def export_to_shapefile_folder(gdf, base_path):
    """
    Export GeoDataFrame to shapefile and organize all related files in a folder.
    Args:
        gdf: GeoDataFrame to export
        base_path: Base path for the output folder (without extension)
    """
    # Ensure CRS is set
    if gdf.crs is None:
        gdf.set_crs(epsg=6539, inplace=True)
    elif gdf.crs.to_epsg() != 6539:
        gdf = gdf.to_crs(epsg=6539)

    # Create folder if it doesn't exist
    folder_path = base_path
    os.makedirs(folder_path, exist_ok=True)

    # Export to shapefile in temporary location
    temp_path = base_path + '_temp'
    gdf.to_file(temp_path + '.shp')

    # Move all related files to the folder
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        temp_file = temp_path + ext
        if os.path.exists(temp_file):
            target_file = os.path.join(folder_path, os.path.basename(base_path) + ext)
            shutil.move(temp_file, target_file)

def clean_and_validate_geometries(gdf):
    """Clean and validate geometries in the GeoDataFrame"""
    # Make valid any invalid geometries
    gdf.geometry = gdf.geometry.make_valid()
    # Buffer by 0 to fix any remaining issues
    gdf.geometry = gdf.geometry.buffer(0)
    return gdf

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    # Load the GeoJSON
    input_path = "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/sites_pop.geojson"
    gdf = gpd.read_file(input_path)

    # Check if top-level weights sum to 1.0
    top_weights = [
        weights["Adaptability_Index_weight"],
        weights["Solar_Energy_Index_weight"],
        weights["Heat_Hazard_Index_weight"],
        weights["Flood_Hazard_Index_weight"],
        weights["Heat_Vulnerability_Index_weight"],
        weights["Flood_Vulnerability_Index_weight"],
        weights["Service_Population_Index_weight"]
    ]
    total_weight = sum(top_weights)
    if not np.isclose(total_weight, 1.0):
        print("Warning: Top-level weights do not sum to 1.0. The final index may not be in [0,1].")

    # Calculate all indices
    Adaptability_Index = create_sub_index(gdf, weights["Adaptability_Index_components"])
    Solar_Energy_Index = create_sub_index(gdf, weights["Solar_Energy_Index_components"])
    Heat_Hazard_Index = create_sub_index(gdf, weights["Heat_Hazard_Index_components"])
    Flood_Hazard_Index = create_sub_index(gdf, weights["Flood_Hazard_Index_components"], flood_hazard=True)
    Heat_Vulnerability_Index = create_sub_index(gdf, weights["Heat_Vulnerability_Index_components"])
    Flood_Vulnerability_Index = create_sub_index(gdf, weights["Flood_Vulnerability_Index_components"])
    Service_Population_Index = create_sub_index(gdf, weights["Service_Population_Index_components"])

    # Combine all indices into a single suitability index
    Suitability_Index = (
        Adaptability_Index * weights["Adaptability_Index_weight"] +
        Solar_Energy_Index * weights["Solar_Energy_Index_weight"] +
        Heat_Hazard_Index * weights["Heat_Hazard_Index_weight"] +
        Flood_Hazard_Index * weights["Flood_Hazard_Index_weight"] +
        Heat_Vulnerability_Index * weights["Heat_Vulnerability_Index_weight"] +
        Flood_Vulnerability_Index * weights["Flood_Vulnerability_Index_weight"] +
        Service_Population_Index * weights["Service_Population_Index_weight"]
    )

    # Create index_norm by normalizing the Suitability_Index
    index_norm = min_max_normalize(pd.Series(Suitability_Index))

    # Add all computed indices to the GeoDataFrame
    gdf["Adaptability_Index"] = Adaptability_Index
    gdf["Solar_Energy_Index"] = Solar_Energy_Index
    gdf["Heat_Hazard_Index"] = Heat_Hazard_Index
    gdf["Flood_Hazard_Index"] = Flood_Hazard_Index
    gdf["Heat_Vulnerability_Index"] = Heat_Vulnerability_Index
    gdf["Flood_Vulnerability_Index"] = Flood_Vulnerability_Index
    gdf["Service_Population_Index"] = Service_Population_Index
    gdf["Suitability_Index"] = Suitability_Index
    gdf["index_norm"] = index_norm

    # Ensure RH_Priority is string type for consistent comparison
    if 'RH_Priority' in gdf.columns:
        gdf['RH_Priority'] = gdf['RH_Priority'].astype(str)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(input_path)

    # Generate timestamp for output folders
    timestamp = datetime.now().strftime('%Y%m%d')

    # Before saving, ensure the GeoDataFrame has the correct CRS (EPSG:6539)
    if gdf.crs is None:
        gdf.set_crs(epsg=6539, inplace=True)
    elif gdf.crs.to_epsg() != 6539:
        gdf = gdf.to_crs(epsg=6539)

    # Save full dataset
    gdf = clean_and_validate_geometries(gdf)
    full_output_path = input_path.replace("_pop.geojson", "_RH_Index.geojson")
    gdf.to_file(full_output_path, driver="GeoJSON")
    full_shapefile_path = os.path.join(output_dir, f'RH_Analysis_Output_{timestamp}')
    export_to_shapefile_folder(gdf, full_shapefile_path)

    # Filter and save priority sites (maintain CRS)
    priority_sites = gdf[gdf['RH_Priority'] == '1'].copy()
    priority_sites = clean_and_validate_geometries(priority_sites)
    priority_output_path = os.path.join(output_dir, 'RH_Primary_Sites.geojson')
    priority_sites.to_file(priority_output_path, driver="GeoJSON")
    priority_shapefile_path = os.path.join(output_dir, f'RH_Primary_Sites_{timestamp}')
    export_to_shapefile_folder(priority_sites, priority_shapefile_path)

    # Filter and save secondary sites (maintain CRS)
    secondary_sites = gdf[gdf['RH_Priority'] == '2'].copy()
    secondary_sites = clean_and_validate_geometries(secondary_sites)
    secondary_output_path = os.path.join(output_dir, 'RH_Secondary_Sites.geojson')
    secondary_sites.to_file(secondary_output_path, driver="GeoJSON")
    secondary_shapefile_path = os.path.join(output_dir, f'RH_Secondary_Sites_{timestamp}')
    export_to_shapefile_folder(secondary_sites, secondary_shapefile_path)

    # Print summary
    print("\nIndexing and export complete.")
    print(f"Full dataset ({len(gdf)} sites):")
    print(f"- GeoJSON: {full_output_path}")
    print(f"- Shapefile folder: {full_shapefile_path}")
    print(f"\nPriority sites ({len(priority_sites)} sites):")
    print(f"- GeoJSON: {priority_output_path}")
    print(f"- Shapefile folder: {priority_shapefile_path}")
    print(f"\nSecondary sites ({len(secondary_sites)} sites):")
    print(f"- GeoJSON: {secondary_output_path}")
    print(f"- Shapefile folder: {secondary_shapefile_path}")
