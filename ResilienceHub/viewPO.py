import json
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)        # Prevent wrapping

def explore_geojson(filepath):
    """
    Reads a GeoJSON file and displays all fields with their first 5 values
    """
    try:
        # Read the GeoJSON file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert features to DataFrame
        features = data['features']
        df = pd.DataFrame([feature['properties'] for feature in features])
        
        # Get list of all columns
        columns = df.columns.tolist()
        
        print("\nDataset Overview:")
        print(f"Total number of fields: {len(columns)}")
        print(f"Total number of records: {len(df)}")
        
        print("\nFields and their first 5 values:")
        print("-" * 80)
        
        for column in columns:
            print(f"\nField: {column}")
            print("Type:", df[column].dtype)
            print("First 5 values:")
            for idx, value in enumerate(df[column].head(), 1):
                print(f"  {idx}. {value}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None

if __name__ == "__main__":
    filepath = '/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/Input/NYC_PostOffices.geojson'
    
    df = explore_geojson(filepath)
    
    if df is not None:
        print("\nDataFrame Summary:")
        print(df.head())