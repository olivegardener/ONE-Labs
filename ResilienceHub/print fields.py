import json
from typing import Dict, Any

def print_feature_fields(feature: Dict[str, Any], indent: int = 0) -> None:
    """Print all fields in a GeoJSON feature with indentation."""
    # Print properties
    if 'properties' in feature:
        print(" " * indent + "Properties:")
        for key, value in feature['properties'].items():
            print(" " * (indent + 2) + f"{key}: {value}")
    
    # Print geometry type and coordinates
    if 'geometry' in feature:
        print(" " * indent + "Geometry:")
        print(" " * (indent + 2) + f"Type: {feature['geometry']['type']}")
        print(" " * (indent + 2) + "Coordinates: [truncated for readability]")

def read_and_print_geojson(file_path: str) -> None:
    """Read a GeoJSON file and print all fields for each feature."""
    try:
        with open(file_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Print GeoJSON type
        print(f"GeoJSON Type: {geojson_data.get('type', 'Not specified')}\n")
        
        # Handle FeatureCollection
        if geojson_data['type'] == 'FeatureCollection':
            features = geojson_data.get('features', [])
            print(f"Number of features: {len(features)}\n")
            
            for i, feature in enumerate(features):
                print(f"Feature {i + 1}:")
                print_feature_fields(feature, indent=2)
                print()  # Empty line between features
        
        # Handle single Feature
        elif geojson_data['type'] == 'Feature':
            print("Single Feature:")
            print_feature_fields(geojson_data, indent=2)
        
        else:
            print(f"Unknown GeoJSON type: {geojson_data['type']}")
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON")
    except KeyError as e:
        print(f"Error: Missing required field {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/output/preprocessed_sites.geojson"
    read_and_print_geojson(file_path)

