"""
heat_analysis.py
Analyzes heat exposure near the sites of interest.

Modifications:
- BUFFER changed to 2000 feet.
- Added timer to measure and print total execution time.
- Added high-level print statements for progress updates.

Steps:
- Reads in 'sites_solar.geojson' from 'output' as 'sites'
- Reads in 'Landsat9_ThermalComposite_ST_B10_2020-2023.tif' from 'input' as 'heathaz'
- Ensures both 'sites' and 'heathaz' are in EPSG:6539 and at desired resolution
- For each site, create a bounding box using the specified buffer distance
- Extract mean Kelvin value in the bounding box, convert to Fahrenheit (F = (K - 273.15)*9/5 + 32)
- Compute 'heat_mean' as this mean Fahrenheit temperature
- Load entire raster, convert to Fahrenheit, and build a cumulative distribution
- Determine the percentile of each site's 'heat_mean' in that distribution
- Convert percentile to a 0-1 range by dividing by 100, round to two decimal places, and store as 'heat_index'
- Save the results as 'sites_heat.geojson'
"""

import geopandas as gpd
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from pathlib import Path
import multiprocessing as mp
import warnings
import os
import time

# -----------------------------------------------------------------------------
# USER CONFIGURATION - MODIFY THESE PARAMETERS
# -----------------------------------------------------------------------------

RESOLUTION = 10.0  # feet
BUFFER = 2000.0    # feet
TARGET_CRS = 'EPSG:6539'

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'output'
INPUT_DIR = SCRIPT_DIR / 'input'

SITES_FILE = OUTPUT_DIR / 'RH_solar.geojson'
HEAT_FILE = INPUT_DIR / 'Landsat9_ThermalComposite_ST_B10_2020-2023.tif'
OUTPUT_FILE = OUTPUT_DIR / 'sites_heat.geojson'

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

def ensure_crs_vector(gdf, target_crs):
    """Ensure the GeoDataFrame is in the target CRS, reproject if needed."""
    if gdf.crs is None:
        warnings.warn("Vector data has no CRS. Setting to target CRS.")
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def ensure_crs_raster(raster_path, target_crs, resolution):
    """
    Ensure the raster is in the target CRS and at the specified resolution.
    If reprojection is needed, create a temporary file and return that path.
    Otherwise, return the original raster_path.
    """
    with rasterio.open(raster_path) as src:
        same_crs = (src.crs is not None and src.crs.to_string() == target_crs)
        same_res = np.isclose(src.res[0], resolution, atol=0.1)
        if not same_crs or not same_res:
            print("Reprojecting raster to target CRS and resolution...")
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution
            )
            profile = src.meta.copy()
            profile.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            reprojected_path = raster_path.parent / f"reprojected_{raster_path.name}"
            with rasterio.open(reprojected_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
            return reprojected_path
        else:
            return raster_path

def kelvin_to_fahrenheit(K):
    return (K - 273.15) * 9/5 + 32

def load_raster_distribution_f(raster_path):
    """
    Load the entire raster into memory (single band),
    convert to Fahrenheit, and return a sorted 1D array of valid temperature values.
    """
    print("Loading and analyzing full heat raster distribution...")
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
    data_f = kelvin_to_fahrenheit(data)
    # Flatten and remove masked elements (nodata)
    valid = data_f.compressed()
    # Sort values
    sorted_values = np.sort(valid)
    return sorted_values

def percentile_from_distribution(value, distribution):
    """
    Given a value and a sorted distribution array,
    find the percentile of the value within that distribution.
    The percentile is the percentage of distribution values <= value.
    """
    idx = np.searchsorted(distribution, value, side='right')
    percentile = (idx / len(distribution)) * 100.0
    return percentile

def extract_mean_temperature(site, raster_path):
    """
    Extract mean temperature (in Fahrenheit) for the bounding box around a site.
    The bounding box is determined by the site centroid plus/minus BUFFER in each direction.
    """
    geom = site.geometry
    if geom is None or geom.is_empty:
        return np.nan

    centroid = geom.centroid
    xmin = centroid.x - BUFFER
    xmax = centroid.x + BUFFER
    ymin = centroid.y - BUFFER
    ymax = centroid.y + BUFFER
    bbox = box(xmin, ymin, xmax, ymax)

    with rasterio.open(raster_path) as src:
        row_start, col_start = src.index(xmin, ymax)
        row_end, col_end = src.index(xmax, ymin)
        row_start, row_end = sorted([row_start, row_end])
        col_start, col_end = sorted([col_start, col_end])

        # Clip to raster bounds
        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_end = min(row_end, src.height - 1)
        col_end = min(col_end, src.width - 1)

        if row_end < row_start or col_end < col_start:
            return np.nan

        window = rasterio.windows.Window(col_start, row_start, (col_end - col_start + 1), (row_end - row_start + 1))
        data = src.read(1, window=window, masked=True)

        if data.size == 0:
            return np.nan

        data_f = kelvin_to_fahrenheit(data)
        mean_temp = data_f.mean()
        return float(mean_temp)

def process_site(args):
    """Process a single site to extract its mean heat value."""
    site, raster_path = args
    return extract_mean_temperature(site, raster_path)

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()
    print("\n=== Starting Heat Analysis ===")
    print(f"Using BUFFER = {BUFFER} feet and RESOLUTION = {RESOLUTION} feet.")

    # Check and load sites
    if not SITES_FILE.exists():
        raise FileNotFoundError(f"Sites file not found at {SITES_FILE}")
    print("Loading sites data...")
    sites = gpd.read_file(SITES_FILE)
    sites = ensure_crs_vector(sites, TARGET_CRS)
    print(f"Loaded {len(sites)} sites.")

    # Check and prepare raster
    if not HEAT_FILE.exists():
        raise FileNotFoundError(f"Heat raster not found at {HEAT_FILE}")
    heathaz_path = ensure_crs_raster(HEAT_FILE, TARGET_CRS, RESOLUTION)

    # Load raster distribution
    distribution = load_raster_distribution_f(heathaz_path)

    # Process sites in parallel
    print("Extracting mean temperature for each site...")
    sites_list = [(row, heathaz_path) for idx, row in sites.iterrows()]
    cpu_count = mp.cpu_count()
    with mp.Pool(cpu_count - 1) as pool:
        mean_temps = pool.map(process_site, sites_list)

    sites['heat_mean'] = mean_temps
    print("Mean temperatures extracted for all sites.")

    # Compute percentile-based index
    print("Calculating heat_index from percentile distribution...")
    percentiles = [percentile_from_distribution(val, distribution) if np.isfinite(val) else np.nan
                   for val in sites['heat_mean']]
    sites['heat_index'] = [round(p/100, 2) if np.isfinite(p) else np.nan for p in percentiles]

    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    sites.to_file(OUTPUT_FILE, driver='GeoJSON')

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n=== Heat Analysis Complete ===")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total processing time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")