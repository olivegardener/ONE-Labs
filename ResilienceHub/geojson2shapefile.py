import geopandas as gpd
import json
from shapely.geometry import shape
import logging

def geojson_to_shapefile(input_geojson, output_shapefile):
    """
    Convert GeoJSON file to Shapefile format with explicit CRS handling.
    
    Args:
        input_geojson (str): Path to input GeoJSON file
        output_shapefile (str): Path for output Shapefile
    """
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Read GeoJSON
        with open(input_geojson, 'r') as f:
            geojson_data = json.load(f)
        
        # Extract features and convert geometries
        features = geojson_data['features']
        geometries = []
        properties = []
        
        for feat in features:
            try:
                if 'geometry' in feat and feat['geometry'] is not None:
                    geom = shape(feat['geometry'])
                    geometries.append(geom)
                    properties.append(feat.get('properties', {}))
            except Exception as e:
                logging.error(f"Error processing feature: {str(e)}")
                continue
        
        # Create GeoDataFrame with explicit geometry column
        gdf = gpd.GeoDataFrame(
            properties,
            geometry=geometries,
            crs="EPSG:6539"  # Explicitly set the source CRS
        )
        
        # Validate the GeoDataFrame
        logging.info(f"Number of features: {len(gdf)}")
        logging.info(f"Geometry types: {gdf.geometry.type.value_counts()}")
        logging.info(f"CRS: {gdf.crs}")
        
        # Check for invalid geometries
        invalid_geoms = gdf[~gdf.geometry.is_valid]
        if not invalid_geoms.empty:
            logging.warning(f"Found {len(invalid_geoms)} invalid geometries")
            # Try to fix invalid geometries
            gdf.geometry = gdf.geometry.buffer(0)
        
        # Save as Shapefile
        gdf.to_file(output_shapefile)
        logging.info(f"Successfully saved to {output_shapefile}")
        
        return gdf
        
    except Exception as e:
        logging.error(f"Error converting file: {str(e)}")
        raise

if __name__ == "__main__":
    gdf = geojson_to_shapefile(
        "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/sites_RH_Index.geojson",
        "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/RH_Analysis_Output1218/sites_RH_all.shp"
    )