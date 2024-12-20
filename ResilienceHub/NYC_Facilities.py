import json
import pandas as pd
from collections import Counter

def analyze_factype(filepath):
    """
    Reads a GeoJSON file and analyzes unique FACTYPE values
    Returns both unique values and their counts
    """
    try:
        # Read the GeoJSON file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract FACTYPE values from features
        factypes = []
        for feature in data['features']:
            factype = feature['properties'].get('FACTYPE')
            if factype:  # Only add non-None values
                factypes.append(factype)
        
        # Get unique values and their counts
        factype_counts = Counter(factypes)
        
        # Sort by count in descending order
        sorted_factypes = sorted(factype_counts.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        return sorted_factypes
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None
    except KeyError:
        print("Error: Unexpected data structure in GeoJSON")
        return None

if __name__ == "__main__":
    filepath = '/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/Input/NYC_Facilities.geojson'
    
    results = analyze_factype(filepath)
    
    if results:
        print("\nUnique FACTYPE values and their counts:")
        print("-" * 40)
        for factype, count in results:
            print(f"{factype}: {count}")
        print(f"\nTotal unique FACTYPE values: {len(results)}")