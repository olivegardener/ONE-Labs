import requests
import json
import pandas as pd

def download_post_office_data():
    """
    Downloads and processes NYC post office GeoJSON data
    Returns both the raw GeoJSON and a pandas DataFrame of the properties
    """
    url = "https://raw.githubusercontent.com/nycommons/nyc-post-offices/refs/heads/master/data/geojson/nyc-post-offices.geojson"
    
    try:
        # Download the data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the GeoJSON
        data = response.json()
        
        # Convert to DataFrame
        features = data['features']
        properties_list = [feature['properties'] for feature in features]
        df = pd.DataFrame(properties_list)
        
        return data, df
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None, None

if __name__ == "__main__":
    # Download and process the data
    geojson_data, post_offices_df = download_post_office_data()
    
    if post_offices_df is not None:
        # Save the raw GeoJSON
        with open('nyc-post-offices.geojson', 'w') as f:
            json.dump(geojson_data, f)
        
        print(f"Downloaded {len(post_offices_df)} post office locations")
        print("\nFirst few rows of the data:")
        print(post_offices_df.head())