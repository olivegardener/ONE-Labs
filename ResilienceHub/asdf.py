import json
import geopandas as gpd
import os
from pathlib import Path

def split_to_shapefiles(input_path):
    # Read the GeoJSON file using geopandas
    try:
        gdf = gpd.read_file(input_path)
    except Exception as e:
        raise ValueError(f"Error reading GeoJSON file: {str(e)}")
    
    # Get the directory of the input file for saving outputs
    output_dir = os.path.dirname(input_path)
    
    # Group by NAME and create separate shapefiles
    for name, group in gdf.groupby('NAME'):
        # Create safe filename by replacing spaces and special characters
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        output_filename = f"BioHotspot_{safe_name}.shp"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the group as a shapefile
        group.to_file(output_path)
        print(f"Created {output_filename}")

if __name__ == "__main__":
    input_path = "/Users/oliveratwood/Desktop/BioHotspots.geojson"
    try:
        split_to_shapefiles(input_path)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")