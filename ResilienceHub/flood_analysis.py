import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import box
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings
import time

# ======================================================================
# USER-DEFINED PARAMETERS
# ======================================================================
BUFFER_DISTANCE_FT = 2000  # radius in feet around site centroid for neighborhood circle
RESOLUTION = 10.0  # raster resolution in feet
NUM_WORKERS = 4  # number of parallel workers
OUTPUT_FILE = "sites_flood.geojson"
TARGET_CRS = "EPSG:6539"

# ======================================================================
# INPUT FILES (Relative Paths)
# Ensure these files exist in the 'input' and 'output' folders next to this script.
# ======================================================================
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
INPUT_DIR = SCRIPT_DIR / "input"
SITES_FILE = OUTPUT_DIR / "sites_heat.geojson"
FEMA_RASTER = INPUT_DIR / "FEMA_FloodHaz_Raster.tif"     # "Fld_Coast"
STORM_RASTER = INPUT_DIR / "Stormwater2080_Raster.tif"   # "Fld_Storm"

# ======================================================================
# FLOOD VALUE DEFINITIONS
# ======================================================================
# FEMA_FloodHaz_Raster (Fld_Coast):
# 0: No Flooding
# 1: 0.2% annual chance flood (coastal 500-yr)
# 2: 1% annual chance flood (coastal 100-yr)
#
# Stormwater2080_Raster (Fld_Storm):
# 0: no flooding
# 1: shallow flooding
# 2: deep flooding
# 3: tidal flooding
#
# Computed fields:
# Cst_500_in, Cst_500_nr, Cst_100_in, Cst_100_nr
# StrmShl_in, StrmShl_nr, StrmDp_in, StrmDp_nr, StrmTid_in, StrmTid_nr
# ======================================================================
COAST_VALUES = {1: '500', 2: '100'}
STORM_VALUES = {1: 'Shl', 2: 'Dp', 3: 'Tid'}

def ensure_crs_vector(gdf, target_crs):
    """Ensure the GeoDataFrame is in the target CRS, reproject if needed."""
    if gdf.crs is None:
        warnings.warn("Vector data has no CRS. Setting to target CRS.")
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def read_raster_window(raster_path, bbox):
    """
    Read a window from the raster defined by a bounding box.
    Return the array and the transform.
    """
    with rasterio.open(raster_path) as src:
        # Check CRS
        if src.crs is not None and src.crs.to_string() != TARGET_CRS:
            raise ValueError(f"Raster {raster_path} CRS ({src.crs}) does not match {TARGET_CRS}.")
        
        window = src.window(*bbox)
        data = src.read(1, window=window, masked=False)
        transform = src.window_transform(window)
        return data, transform

def process_site(args):
    """
    Process a single site to calculate the flood-related fields.
    Now uses a circle polygon for the neighborhood rather than a bounding box.

    Steps:
    1. Create a circle polygon around the site centroid using BUFFER_DISTANCE_FT.
    2. Determine the bounding box of the circle polygon and read the raster data for that area.
    3. Rasterize both the site polygon (for _in calculations) and the circle polygon (for _nr calculations).
    4. Compute flood value fractions inside the site polygon (original footprint) and inside the circle polygon.
    """
    idx, site, coast_path, storm_path, buffer_dist = args
    geom = site.geometry
    if geom is None or geom.is_empty:
        # Return zeros for all fields if invalid geometry
        return idx, {col: 0.0 for col in [
            'Cst_500_in','Cst_500_nr','Cst_100_in','Cst_100_nr',
            'StrmShl_in','StrmShl_nr','StrmDp_in','StrmDp_nr','StrmTid_in','StrmTid_nr'
        ]}

    # Create circle polygon around centroid
    centroid = geom.centroid
    circle_geom = centroid.buffer(buffer_dist)  # circle polygon

    # Get bounding box from the circle polygon
    minx, miny, maxx, maxy = circle_geom.bounds
    bbox = (minx, miny, maxx, maxy)

    # Read raster windows
    coast_arr, coast_transform = read_raster_window(coast_path, bbox)
    storm_arr, storm_transform = read_raster_window(storm_path, bbox)

    # Ensure both arrays are same shape
    min_height = min(coast_arr.shape[0], storm_arr.shape[0])
    min_width = min(coast_arr.shape[1], storm_arr.shape[1])
    coast_arr = coast_arr[:min_height, :min_width]
    storm_arr = storm_arr[:min_height, :min_width]

    width = coast_arr.shape[1]
    height = coast_arr.shape[0]

    # Rasterize site polygon (for _in calculations)
    site_rast = features.rasterize(
        [(geom, 1)],
        out_shape=(height, width),
        transform=coast_transform,
        fill=0,
        dtype=np.uint8
    )

    # Rasterize circle polygon (for _nr calculations)
    circle_rast = features.rasterize(
        [(circle_geom, 1)],
        out_shape=(height, width),
        transform=coast_transform,
        fill=0,
        dtype=np.uint8
    )

    site_mask = (site_rast == 1)
    circle_mask = (circle_rast == 1)

    inside_count = site_mask.sum()       # pixels inside site polygon
    circle_count = circle_mask.sum()     # pixels inside the circle polygon (neighborhood)

    results = {}

    # Inside polygon fractions (_in)
    if inside_count == 0:
        for cval in COAST_VALUES.values():
            results[f"Cst_{cval}_in"] = 0.0
        for sval in STORM_VALUES.values():
            results[f"Strm{sval}_in"] = 0.0
    else:
        # Coast inside polygon
        for cval, ctag in COAST_VALUES.items():
            inside_match = ((site_mask) & (coast_arr == cval)).sum()
            results[f"Cst_{ctag}_in"] = inside_match / inside_count

        # Storm inside polygon
        for sval, stag in STORM_VALUES.items():
            inside_match = ((site_mask) & (storm_arr == sval)).sum()
            results[f"Strm{stag}_in"] = inside_match / inside_count

    # Neighborhood fractions (_nr) using circle polygon
    if circle_count == 0:
        for cval in COAST_VALUES.values():
            results[f"Cst_{cval}_nr"] = 0.0
        for sval in STORM_VALUES.values():
            results[f"Strm{sval}_nr"] = 0.0
    else:
        # Coast in neighborhood (circle)
        for cval, ctag in COAST_VALUES.items():
            nr_match = ((circle_mask) & (coast_arr == cval)).sum()
            results[f"Cst_{ctag}_nr"] = nr_match / circle_count

        # Storm in neighborhood (circle)
        for sval, stag in STORM_VALUES.items():
            nr_match = ((circle_mask) & (storm_arr == sval)).sum()
            results[f"Strm{stag}_nr"] = nr_match / circle_count

    return idx, results

def main():
    start_time = time.time()

    # Load sites
    if not SITES_FILE.exists():
        raise FileNotFoundError(f"Sites file not found: {SITES_FILE}")

    sites = gpd.read_file(SITES_FILE)
    sites = ensure_crs_vector(sites, TARGET_CRS)

    # Prepare output columns
    output_cols = [
        'Cst_500_in','Cst_500_nr','Cst_100_in','Cst_100_nr',
        'StrmShl_in','StrmShl_nr','StrmDp_in','StrmDp_nr','StrmTid_in','StrmTid_nr'
    ]
    for col in output_cols:
        if col not in sites.columns:
            sites[col] = 0.0

    # Parallel processing
    args_list = [(idx, row, FEMA_RASTER, STORM_RASTER, BUFFER_DISTANCE_FT) for idx, row in sites.iterrows()]

    print("Starting parallel flood analysis...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for fut in executor.map(process_site, args_list):
            idx, results = fut
            for k, v in results.items():
                sites.at[idx, k] = v

    # Save results
    out_path = OUTPUT_DIR / OUTPUT_FILE
    sites.to_file(out_path, driver='GeoJSON')
    print(f"Results saved to {out_path}")

    elapsed = time.time() - start_time
    print(f"Flood analysis completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

if __name__ == "__main__":
    main()