import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime
import os
import shutil
from pathlib import Path

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------
def export_to_shapefile_folder(gdf, base_path):
    """
    Export GeoDataFrame to shapefile and organize all related files in a folder.
    """
    # Ensure CRS is set to EPSG:6539
    if gdf.crs is None:
        gdf.set_crs(epsg=6539, inplace=True)
    elif gdf.crs.to_epsg() != 6539:
        gdf = gdf.to_crs(epsg=6539)

    folder_path = base_path
    os.makedirs(folder_path, exist_ok=True)

    temp_path = base_path + '_temp'
    gdf.to_file(temp_path + '.shp')

    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        temp_file = temp_path + ext
        if os.path.exists(temp_file):
            target_file = os.path.join(folder_path, os.path.basename(base_path) + ext)
            shutil.move(temp_file, target_file)

def clean_and_validate_geometries(gdf):
    """Clean and validate geometries in the GeoDataFrame."""
    gdf.geometry = gdf.geometry.make_valid()
    gdf.geometry = gdf.geometry.buffer(0)
    return gdf

# --------------------------------------------------------------------
# Main Preprocessing Script
# --------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    
    # 1) Use a single Path object, not a set:
    input_path = script_dir / "output" / "sites_pop.geojson"
    print(f"Reading input from: {input_path}")
    
    # 2) Verify the file actually exists:
    if not input_path.exists():
        raise FileNotFoundError(f"File not found at: {input_path}")

    gdf = gpd.read_file(input_path)

    # (Then continue with your cleaning and saving steps)
    if gdf.crs is None:
        gdf.set_crs(epsg=6539, inplace=True)
    elif gdf.crs.to_epsg() != 6539:
        gdf = gdf.to_crs(epsg=6539)

    gdf = clean_and_validate_geometries(gdf)

    if 'RH_Priority' in gdf.columns:
        gdf['RH_Priority'] = gdf['RH_Priority'].astype(str)

    # Example: save entire dataset after cleaning
    full_output_path = str(input_path).replace("_pop.geojson", "_cleaned.geojson")
    gdf.to_file(full_output_path, driver="GeoJSON")

    output_dir = os.path.dirname(input_path)
    timestamp = datetime.now().strftime('%Y%m%d')
    full_shapefile_path = os.path.join(output_dir, f'RH_Analysis_Output_{timestamp}')
    export_to_shapefile_folder(gdf, full_shapefile_path)

    # Split into Priority=1 and Priority=2 subsets
    priority_sites = gdf[gdf['RH_Priority'] == '1'].copy()
    priority_sites = clean_and_validate_geometries(priority_sites)
    priority_output_path = os.path.join(output_dir, 'RH_Primary_Sites.geojson')
    priority_sites.to_file(priority_output_path, driver="GeoJSON")
    priority_shapefile_path = os.path.join(output_dir, f'RH_Primary_Sites_{timestamp}')
    export_to_shapefile_folder(priority_sites, priority_shapefile_path)

    secondary_sites = gdf[gdf['RH_Priority'] == '2'].copy()
    secondary_sites = clean_and_validate_geometries(secondary_sites)
    secondary_output_path = os.path.join(output_dir, 'RH_Secondary_Sites.geojson')
    secondary_sites.to_file(secondary_output_path, driver="GeoJSON")
    secondary_shapefile_path = os.path.join(output_dir, f'RH_Secondary_Sites_{timestamp}')
    export_to_shapefile_folder(secondary_sites, secondary_shapefile_path)

    print("\nPreprocessing complete.")
    print(f"- Full cleaned dataset: {full_output_path}")
    print(f"  Shapefile folder: {full_shapefile_path}")
    print(f"- Primary sites: {priority_output_path}")
    print(f"  Shapefile folder: {priority_shapefile_path}")
    print(f"- Secondary sites: {secondary_output_path}")
    print(f"  Shapefile folder: {secondary_shapefile_path}")