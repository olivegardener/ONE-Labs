# Imports
import geopandas as gpd
import pandas as pd
import numpy as np
import numba
from numba import njit
import logging
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pvlib
import multiprocessing as mp  # For cpu_count and Pool
from multiprocessing import Pool, cpu_count
from datetime import datetime
import fiona
from pathlib import Path
import time  # Added import for timing
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shadow Optimization
@numba.njit
def calculate_shadow_impact_numba(height_diff_array, sun_altitude, distance_array):
    """Calculate the shadow impact for arrays of buildings."""
    impacts = np.zeros_like(height_diff_array)
    tan_sun_altitude = np.tan(np.radians(sun_altitude))
    for i in range(len(height_diff_array)):
        height_diff = height_diff_array[i]
        distance = distance_array[i]

        if sun_altitude <= 0:
            impacts[i] = 1.0
            continue
        if height_diff <= 0:
            impacts[i] = 0.0
            continue

        shadow_length = height_diff / tan_sun_altitude
        if distance <= shadow_length:
            impacts[i] = 1.0 - (distance / shadow_length)
        else:
            impacts[i] = 0.0

    return impacts

# Shadow Vectorization Definition
def calculate_shadow_impact_vectorized(building_row, buildings_gdf, solar_pos, spatial_index):
    """Calculate shadow impact for a building considering nearby buildings."""
    try:
        MAX_SHADOW_DISTANCE = 3 * building_row['heightroof']
        bounds = building_row.geometry.bounds

        # Expand bounds by MAX_SHADOW_DISTANCE
        bounds = (
            bounds[0] - MAX_SHADOW_DISTANCE,
            bounds[1] - MAX_SHADOW_DISTANCE,
            bounds[2] + MAX_SHADOW_DISTANCE,
            bounds[3] + MAX_SHADOW_DISTANCE
        )

        # Use spatial index to find nearby buildings
        possible_matches_idx = list(spatial_index.intersection(bounds))
        nearby_buildings = buildings_gdf.iloc[possible_matches_idx]
        nearby_buildings = nearby_buildings[nearby_buildings.index != building_row.name]

        if nearby_buildings.empty:
            return 1.0

        ref_centroid = building_row.geometry.centroid.coords[0]
        nearby_centroids = np.array([geom.centroid.coords[0] for geom in nearby_buildings.geometry])

        dx = nearby_centroids[:, 0] - ref_centroid[0]
        dy = nearby_centroids[:, 1] - ref_centroid[1]
        distances = np.sqrt(dx**2 + dy**2)

        height_diff = nearby_buildings['heightroof'].values - building_row['heightroof']
        sun_altitude = solar_pos['apparent_elevation']

        shadow_impacts = calculate_shadow_impact_numba(
            height_diff,
            sun_altitude,
            distances
        )

        total_shadow_impact = np.mean(shadow_impacts)
        shadow_factor = max(0.0, 1.0 - total_shadow_impact)
        return shadow_factor

    except Exception as e:
        logger.error(f"Error calculating shadow impact for building {building_row.name}: {str(e)}")
        return 1.0

# Global variables for worker processes
global_buildings_gdf = None
global_spatial_index = None
global_solar_position = None
global_annual_radiation = None
global_panel_density = None
global_panel_efficiency = None
global_performance_ratio = None

def init_worker(buildings_gdf, spatial_index, solar_position, annual_radiation, panel_density, panel_efficiency, performance_ratio):
    """Initialize global variables in worker processes."""
    global global_buildings_gdf
    global global_spatial_index
    global global_solar_position
    global global_annual_radiation
    global global_panel_density
    global global_panel_efficiency
    global global_performance_ratio

    global_buildings_gdf = buildings_gdf
    global_spatial_index = spatial_index
    global_solar_position = solar_position
    global_annual_radiation = annual_radiation
    global_panel_density = panel_density
    global_panel_efficiency = panel_efficiency
    global_performance_ratio = performance_ratio

# Worker function at the module level
def worker_process_building(args):
    """Worker function to process a single building using global variables."""
    idx, building = args
    area_m2 = building['Shape_Area'] * 0.092903  # Convert from square feet to square meters

    # Access global variables
    buildings_gdf = global_buildings_gdf
    spatial_index = global_spatial_index
    solar_pos = global_solar_position
    annual_radiation = global_annual_radiation
    panel_density = global_panel_density
    panel_efficiency = global_panel_efficiency
    performance_ratio = global_performance_ratio

    # Calculate the shadow factor
    shadow_factor = calculate_shadow_impact_vectorized(
        building, buildings_gdf, solar_pos, spatial_index
    )

    solar_potential = (
        annual_radiation *
        area_m2 *
        panel_density *
        panel_efficiency *
        performance_ratio *
        shadow_factor
    )

    return {
        'solar_potential': float(solar_potential),
        'effective_area': float(area_m2 * panel_density),
        'peak_power': float(area_m2 * panel_density * panel_efficiency),
        'shadow_factor': float(shadow_factor),
        'annual_radiation': float(annual_radiation),
        'performance_ratio': float(performance_ratio)
    }

# SolarAnalyzer Class Definition
class SolarAnalyzer:
    def __init__(self):
        logger.info("Initializing SolarAnalyzer...")
        self._initialize_constants()
        self._initialize_cache()
        self.buildings_gdf = None

    def _initialize_constants(self):
        self.NYC_LAT = 40.7128
        self.NYC_LON = -74.0060
        self.PANEL_EFFICIENCY = 0.20
        self.PERFORMANCE_RATIO = 0.75
        self.PANEL_DENSITY = 0.70
        self._solar_position = None
        self._spatial_index = None

    def _initialize_cache(self):
        self._monthly_radiation = self._initialize_radiation_data()
        self._annual_radiation = self._calculate_annual_radiation()

    def _initialize_radiation_data(self):
        """Initialize default radiation data (kWh/m²/day)."""
        return {
            '01': 2.45, '02': 3.42, '03': 4.53, '04': 5.64,
            '05': 6.48, '06': 6.89, '07': 6.75, '08': 5.98,
            '09': 4.92, '10': 3.67, '11': 2.56, '12': 2.12
        }

    def _calculate_annual_radiation(self):
        """Calculate annual radiation from monthly data (kWh/m²/year)."""
        monthly_sum = sum(self._monthly_radiation.values())
        return monthly_sum * 365 / 12

    def initialize_spatial_index(self, buildings_gdf):
        """Initialize spatial index for buildings."""
        self.spatial_index = buildings_gdf.sindex
        return buildings_gdf

    def get_solar_position(self):
        """Calculate solar position at a specific time."""
        if self._solar_position is None:
            times = pd.date_range('2020-06-21 12:00:00', periods=1, freq='H', tz='UTC')  # Summer solstice at noon
            location = pvlib.location.Location(latitude=self.NYC_LAT, longitude=self.NYC_LON)
            self._solar_position = location.get_solarposition(times).iloc[0]
        return self._solar_position

# Function for processing buildings in parallel
def process_buildings_parallel(analyzer, buildings_chunk, buildings_gdf):
    """Process buildings using multiprocessing Pool without progress bar."""
    analyzer.buildings_gdf = buildings_gdf  # Store for access in worker functions
    analyzer.initialize_spatial_index(buildings_gdf)  # Ensure spatial index is initialized

    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Using {num_processes} processes for parallel processing.")

    # Prepare the list of arguments for the worker function
    args_list = [(idx, row) for idx, row in buildings_chunk.iterrows()]

    # Prepare data for workers
    solar_position = analyzer.get_solar_position()
    spatial_index = analyzer.spatial_index
    annual_radiation = analyzer._annual_radiation
    panel_density = analyzer.PANEL_DENSITY
    panel_efficiency = analyzer.PANEL_EFFICIENCY
    performance_ratio = analyzer.PERFORMANCE_RATIO

    # Use the standard multiprocessing Pool with initializer
    with Pool(processes=num_processes, initializer=init_worker, initargs=(
        buildings_gdf, spatial_index, solar_position, annual_radiation,
        panel_density, panel_efficiency, performance_ratio
    )) as pool:
        # Map the worker function over the list of arguments
        results_list = pool.map(worker_process_building, args_list)

    # Create DataFrame from results
    results_df = pd.DataFrame(results_list, index=buildings_chunk.index)

    return results_df

# Function to analyze solar potential
def analyze_solar_potential(candidate_buildings, full_buildings):
    """
    Main function to analyze solar potential for candidate buildings.
    """
    try:
        # Initialize analyzer
        analyzer = SolarAnalyzer()
        logger.info("Initialized SolarAnalyzer")

        # Validate CRS
        if candidate_buildings.crs is None or full_buildings.crs is None:
            logger.warning("Input data missing CRS, assuming EPSG:4326")
            candidate_buildings = candidate_buildings.set_crs(epsg=4326)
            full_buildings = full_buildings.set_crs(epsg=4326)

        # Project to a suitable CRS for distance calculations (e.g., EPSG:2263)
        candidate_projected = candidate_buildings.to_crs(epsg=2263)
        buildings_projected = full_buildings.to_crs(epsg=2263)

        # Initialize spatial index
        analyzer.initialize_spatial_index(buildings_projected)

        # Process buildings using parallel processing
        results = process_buildings_parallel(
            analyzer,
            candidate_projected,
            buildings_projected
        )

        # Merge results back to original GeoDataFrame
        analyzed_buildings = candidate_projected.join(results)

        # Reproject back to original CRS
        return analyzed_buildings.to_crs(candidate_buildings.crs)

    except Exception as e:
        logger.error(f"Error in solar potential analysis: {str(e)}")
        return None

# Prepare and Process Candidate Buildings
def filter_pois(pois):
    target_classes = [
        'arts_centre', 'college', 'community_centre',
        'kindergarten', 'library', 'school', 'shelter'
    ]
    filtered_pois = pois[pois['fclass'].isin(target_classes)].copy()
    print(f"\nFiltered POIs by class:")
    print(filtered_pois['fclass'].value_counts())
    return filtered_pois

def prepare_firehouses(firehouses):
    # Create a copy of the firehouses dataset
    firehouses_prep = firehouses.copy()

    # Add required columns with appropriate values
    firehouses_prep['fclass'] = 'Firehouse'  # Just the text 'Firehouse'
    firehouses_prep['name'] = firehouses_prep['FacilityName']  # Use FacilityName as name

    # Select only needed columns
    cols_to_keep = ['fclass', 'name', 'geometry']
    firehouses_prep = firehouses_prep[cols_to_keep]

    print(f"\nPrepared firehouses:")
    print(f"Total firehouses: {len(firehouses_prep)}")
    return firehouses_prep

def merge_candidates(filtered_pois, pofw, firehouses):
    # Prepare POFW by adding prefix to fclass and ensuring it has a name column
    pofw_prep = pofw.copy()
    pofw_prep['fclass'] = 'pofw_' + pofw_prep['fclass']
    if 'name' not in pofw_prep.columns:
        pofw_prep['name'] = pofw_prep.get('Name', 'Unknown')  # Adjust field name as needed

    # Prepare firehouses
    firehouses_prep = prepare_firehouses(firehouses)

    # Ensure filtered_pois has a name column
    if 'name' not in filtered_pois.columns:
        filtered_pois['name'] = filtered_pois.get('Name', 'Unknown')  # Adjust field name as needed

    # Select columns to keep
    cols_to_keep = ['fclass', 'name', 'geometry']

    # Verify all dataframes have required columns before concat
    for df, df_name in [(filtered_pois, 'filtered_pois'), 
                        (pofw_prep, 'pofw_prep'), 
                        (firehouses_prep, 'firehouses_prep')]:
        missing_cols = set(cols_to_keep) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} in {df_name}")

    # Merge all three datasets
    candidates = pd.concat([
        filtered_pois[cols_to_keep],
        pofw_prep[cols_to_keep],
        firehouses_prep[cols_to_keep]
    ], ignore_index=True)

    # Add ObjectID
    candidates['ObjectID'] = candidates.index + 1

    print(f"\nMerged candidates by class:")
    print(candidates['fclass'].value_counts())
    return candidates


def load_gdb_layers(gdb_path):
    # List all layers in the geodatabase
    layers = fiona.listlayers(gdb_path)
    print(f"Available layers in geodatabase: {layers}")

    # Load required layers
    buildings = gpd.read_file(gdb_path, layer='NYC_Buildings')
    pois = gpd.read_file(gdb_path, layer='NYC_POIS')
    pofw = gpd.read_file(gdb_path, layer='NYC_POFW')

    return buildings, pois, pofw

def combine_overlapping_points(buildings, candidate_points):
    # Perform spatial join but keep all matches
    joined = gpd.sjoin(
        buildings,
        candidate_points,
        how='inner',
        predicate='contains'
    )
    
    # Print joined columns for debugging
    print("\nJoined columns after sjoin:")
    print(joined.columns.tolist())
    
    # Use the correct column names without resetting the index
    grouped = joined.groupby(joined.index).agg({
        'fclass': lambda x: ' + '.join(sorted(set(x))),       # 'fclass' is unsuffixed
        'name_right': lambda x: ' + '.join(sorted(set(x))),   # 'name_right' from candidate_points
    })
    
    # The index of 'grouped' corresponds to the index of 'buildings'
    # So we can assign directly
    result = buildings.loc[grouped.index].copy()
    result['fclass'] = grouped['fclass']
    result['name'] = grouped['name_right']
    
    return result

# Prepare Parking Data
def prepare_parking(parking):
    # Create a copy of the parking dataset
    parking_prep = parking.copy()

    # Add required columns with appropriate values
    parking_prep['fclass'] = 'City-Owned Parking Lot'  # Set fclass for all rows
    parking_prep['name'] = parking_prep['Address']  # Use Address as name

    # Select only needed columns
    cols_to_keep = ['fclass', 'name', 'geometry']
    parking_prep = parking_prep[cols_to_keep]

    print(f"\nPrepared parking data:")
    print(f"Total parking lots: {len(parking_prep)}")
    return parking_prep

# Combine Parking Data with Candidate Buildings
def merge_parking_with_candidates(candidate_buildings, parking_prep):
    # Ensure that the parking data has the same CRS as the candidate buildings
    if parking_prep.crs != candidate_buildings.crs:
        parking_prep = parking_prep.to_crs(candidate_buildings.crs)

    # Concatenate the parking data with the candidate buildings
    combined_candidates = pd.concat([candidate_buildings, parking_prep], ignore_index=True)

    # Reset index to maintain consistency
    combined_candidates.reset_index(drop=True, inplace=True)

    return combined_candidates

if __name__ == "__main__":
    # Start the overall timer
    script_start_time = time.time()

    # Load data
    # Set base input directory
    INPUT_DIR = '/Users/oliveratwood/One Architecture Dropbox/Oliver Atwood/P2415_CSC Year Two/05 GIS/06 Scripts/ResilienceHub/Input'

    # Derive paths for input files
    gdb_path = f"{INPUT_DIR}/ResilienceHub.gdb"
    firehouse_path = f"{INPUT_DIR}/FDNY_Firehouse.geojson"
    parking_path = f"{INPUT_DIR}/Parking_City.geojson"

    # Load the data
    buildings, pois, pofw = load_gdb_layers(gdb_path)
    firehouses = gpd.read_file(firehouse_path)
    parking = gpd.read_file(parking_path)

    print("\nDataset Summary:")
    print(f"Number of buildings: {len(buildings):,}")
    print(f"Number of Places of Interest: {len(pois):,}")
    print(f"Number of Places of Worship: {len(pofw):,}")
    print(f"Number of Firehouses: {len(firehouses):,}")
    print(f"Number of City-Owned Parking Lots: {len(parking):,}")

    # Display sample of each dataset
    print("\nBuildings sample columns:")
    print(buildings.columns.tolist())
    print("\nPOIs sample columns:")
    print(pois.columns.tolist())

    # Process candidates
    filtered_pois = filter_pois(pois)
    candidate_points = merge_candidates(filtered_pois, pofw, firehouses)

    print("\nVerifying candidate points structure:")
    print("Columns:", candidate_points.columns.tolist())
    print("\nSample of candidate points:")
    print(candidate_points[['fclass', 'name']].head())

    # Spatial join to get candidate buildings
    candidate_buildings = combine_overlapping_points(buildings, candidate_points)

    # Prepare parking data
    parking_prep = prepare_parking(parking)

    # Merge parking data with candidate buildings
    candidate_buildings = merge_parking_with_candidates(candidate_buildings, parking_prep)

    # Check for NaN values in 'fclass'
    num_nan = candidate_buildings['fclass'].isna().sum()
    total = len(candidate_buildings)
    print(f"\nNumber of NaN values in 'fclass': {num_nan} out of {total}")

    # Print value counts including NaN
    print("\nDistribution by facility type (including NaN):")
    print(candidate_buildings['fclass'].value_counts(dropna=False))

    print(f"\nFinal candidate buildings including parking lots:")
    print(f"Total buildings selected: {len(candidate_buildings):,}")
    print("\nDistribution by facility type:")
    print(candidate_buildings['fclass'].value_counts())

    multi_use = candidate_buildings[candidate_buildings['fclass'].str.contains('\+')]

    print(f"\nBuildings with multiple uses: {len(multi_use)}")
    print("\nSample of multi-use buildings:")
    print(multi_use[['fclass', 'name']].head())

    # Ensure required columns are present
    required_columns = ['Shape_Area', 'heightroof', 'geometry']
    missing_cols = [col for col in required_columns if col not in candidate_buildings.columns]
    if missing_cols:
        print(f"Adding missing columns: {missing_cols}")
        for col in missing_cols:
            if col == 'Shape_Area':
                # Calculate area in square meters (assuming projected CRS)
                candidate_buildings[col] = candidate_buildings['geometry'].area
            elif col == 'heightroof':
                # Assign default height for parking lots if needed (e.g., 0 meters)
                candidate_buildings[col] = candidate_buildings.get('heightroof', 0)
            else:
                candidate_buildings[col] = None  # Assign None or appropriate default

    # # Sample candidate buildings for testing
    # sample_size = 50
    # candidate_buildings_sample = candidate_buildings.sample(sample_size, random_state=42)

    # Start timer for solar potential analysis
    analysis_start_time = time.time()

    # Analyze solar potential with parallel processing
    analyzed_buildings = analyze_solar_potential(candidate_buildings, buildings)

    # End timer for solar potential analysis
    analysis_end_time = time.time()
    analysis_duration = analysis_end_time - analysis_start_time
    print(f"\nSolar potential analysis completed in {analysis_duration:.2f} seconds.")

    # Check results
    if analyzed_buildings is not None and not analyzed_buildings.empty:
        print("\nSolar analysis completed successfully.")
        print(f"Number of buildings analyzed: {len(analyzed_buildings)}")

        # Display summary statistics
        print("\nSummary of solar potential:")
        print(analyzed_buildings['solar_potential'].describe())

        # Display shadow factor statistics
        print("\nShadow Factor Statistics:")
        print(analyzed_buildings['shadow_factor'].describe())

        #export analysis
        output_dir = Path('results')
        output_file = output_dir / 'analyzed_buildings.geojson'
        analyzed_buildings.to_file(output_file, driver='GeoJSON')

    else:
        print("Solar analysis returned no results.")

    # End the overall timer
    script_end_time = time.time()
    script_duration = script_end_time - script_start_time
    print(f"\nTotal script execution time: {script_duration/60:.2f} minutes")
