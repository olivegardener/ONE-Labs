import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import warnings
from shapely.geometry import Point
from pathlib import Path
import time

# User-defined parameters
BUFFER = 2000.0  # feet
TARGET_CRS = 'EPSG:6539'  # Projected CRS in feet
CENSUS_API_KEY = "a3ebdf1648b7fb21df55df7246d9642f040c0ee0"  # Replace with your actual key

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
INPUT_DIR = SCRIPT_DIR / "input"
SITES_FILE = OUTPUT_DIR / "sites_vul.geojson"
OUTPUT_FILE = "sites_pop.geojson"

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_36_tabblock20.zip"

# NYC county FIPS codes
nyc_counties = ['005','047','061','081','085']

def ensure_crs_vector(gdf, target_crs):
    if gdf.crs is None:
        warnings.warn("Vector data has no CRS. Setting to target CRS.")
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def download_and_extract_tiger(url, extract_dir):
    extract_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=extract_dir)
    shp_path = extract_dir / "tl_2020_36_tabblock20.shp"
    if not shp_path.exists():
        raise FileNotFoundError("Shapefile not found after extraction.")
    return shp_path

def fetch_block_population_for_county(state_fips, county_fips, api_key):
    base_url = "https://api.census.gov/data/2020/dec/pl"
    params = {
        "get": "P1_001N",
        "for": "block:*",
        "in": f"state:{state_fips} county:{county_fips}",
        "key": api_key
    }
    r = requests.get(base_url, params=params)
    r.raise_for_status()

    data = r.json()
    if len(data) <= 1:
        # No data returned for this county
        return pd.DataFrame(columns=["GEOID20","P1_001N"])

    headers = data[0]
    values = data[1:]
    if not all(col in headers for col in ['P1_001N','state','county','tract','block']):
        # Unexpected response format
        return pd.DataFrame(columns=["GEOID20","P1_001N"])

    df = pd.DataFrame(values, columns=headers)
    df['GEOID20'] = df['state'] + df['county'] + df['tract'] + df['block']
    df['P1_001N'] = pd.to_numeric(df['P1_001N'], errors='coerce').fillna(0).astype(int)
    return df[['GEOID20','P1_001N']]

def main():
    start_time = time.time()

    print("Extracting Census Data...")

    if not SITES_FILE.exists():
        raise FileNotFoundError(f"Sites file not found: {SITES_FILE}")

    tiger_dir = INPUT_DIR / "tiger_blocks"
    shp_path = tiger_dir / "tl_2020_36_tabblock20.shp"
    if not shp_path.exists():
        shp_path = download_and_extract_tiger(TIGER_URL, tiger_dir)

    sites = gpd.read_file(SITES_FILE)
    sites = ensure_crs_vector(sites, TARGET_CRS)

    # Load NYC blocks
    blocks = gpd.read_file(str(shp_path))
    blocks = blocks[blocks['GEOID20'].str[2:5].isin(nyc_counties)]
    blocks = ensure_crs_vector(blocks, TARGET_CRS)

    # Fetch population data for NYC counties
    pop_dfs = []
    for c in nyc_counties:
        df_county = fetch_block_population_for_county('36', c, CENSUS_API_KEY)
        if not df_county.empty:
            pop_dfs.append(df_county)
    if not pop_dfs:
        raise ValueError("No population data retrieved from Census API. Check API key and parameters.")
    pop_data = pd.concat(pop_dfs, ignore_index=True)

    # Merge population with blocks
    blocks = blocks.merge(pop_data, on='GEOID20', how='left')
    blocks['P1_001N'] = blocks['P1_001N'].fillna(0)

    block_sindex = blocks.sindex

    if 'pop_est' not in sites.columns:
        sites['pop_est'] = np.nan

    for idx, site in sites.iterrows():
        geom = site.geometry
        if geom is None or geom.is_empty:
            sites.at[idx, 'pop_est'] = np.nan
            continue

        centroid = geom.centroid
        buffer_geom = centroid.buffer(BUFFER)

        candidate_idxs = list(block_sindex.intersection(buffer_geom.bounds))
        candidate_blocks = blocks.iloc[candidate_idxs].copy()
        if candidate_blocks.empty:
            sites.at[idx, 'pop_est'] = 0.0
            continue

        buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_geom], crs=sites.crs)
        intersection = gpd.overlay(candidate_blocks, buffer_gdf, how='intersection')

        if intersection.empty:
            sites.at[idx, 'pop_est'] = 0.0
            continue

        intersection['intersect_area'] = intersection.geometry.area
        candidate_blocks['block_area'] = candidate_blocks.geometry.area

        # Use suffixes to avoid column conflicts
        intersection = intersection.merge(candidate_blocks[['GEOID20','block_area','P1_001N']],
                                          on='GEOID20', how='left', suffixes=('', '_block'))

        if 'P1_001N_block' not in intersection.columns:
            raise KeyError("P1_001N missing after merge. Check data availability.")

        intersection['proportion'] = intersection['intersect_area'] / intersection['block_area']
        intersection['weighted_pop'] = intersection['proportion'] * intersection['P1_001N_block']

        total_pop = intersection['weighted_pop'].sum()
        sites.at[idx, 'pop_est'] = round(total_pop)

    out_path = OUTPUT_DIR / OUTPUT_FILE
    sites.to_file(out_path, driver='GeoJSON')
    print(f"Results saved to {out_path}")

    elapsed = time.time() - start_time
    print(f"Census data analysis completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

if __name__ == "__main__":
    main()