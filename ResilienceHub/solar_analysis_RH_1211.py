"""
solar_analysis.py
Updated to:
- Load preprocessed_sites.geojson
- Convert heightroof to a numeric value and fill missing
- Analyze solar potential directly.

This script:
- Loads preprocessed sites from ./output/preprocessed_sites.geojson
- Performs solar potential analysis on these sites
- Outputs the result as ./output/sites_solar.geojson and ./output/shapefiles/sites_solar.shp
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import numba
from numba import njit
import logging
import pvlib
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import warnings
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@numba.njit
def calculate_shadow_impact_numba(height_diff_array, sun_altitude, distance_array):
    """
    Calculate the shadow impact for arrays of buildings.

    Args:
        height_diff_array (np.array): Array of height differences
        sun_altitude (float): Solar altitude angle in degrees
        distance_array (np.array): Distances between buildings

    Returns:
        np.array: Array of shadow impact values between 0 and 1
    """
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

def calculate_shadow_impact_vectorized(building_row, buildings_gdf, solar_pos, spatial_index):
    """
    Calculate shadow impact for a building considering nearby buildings.

    Args:
        building_row (GeoSeries): Row containing building data
        buildings_gdf (GeoDataFrame): GeoDataFrame containing all buildings
        solar_pos (Series): Solar position data
        spatial_index: Spatial index for buildings

    Returns:
        float: Shadow factor between 0 and 1
    """
    try:
        # Set maximum distance to check for shadows
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

def init_worker(buildings_gdf, spatial_index, solar_position, annual_radiation,
                panel_density, panel_efficiency, performance_ratio):
    """
    Initialize global variables in worker processes.
    """
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

def worker_process_building(args):
    """
    Worker function to process a single building using global variables.
    """
    idx, building = args
    # Assume CRS is in feet and convert area to square meters
    area_ft2 = building.geometry.area if 'geometry' in building else 0
    area_m2 = area_ft2 * 0.092903

    buildings_gdf = global_buildings_gdf
    spatial_index = global_spatial_index
    solar_pos = global_solar_position
    annual_radiation = global_annual_radiation
    panel_density = global_panel_density
    panel_efficiency = global_panel_efficiency
    performance_ratio = global_performance_ratio

    # Calculate shadow factor
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

class SolarAnalyzer:
    """Class for analyzing solar potential of buildings."""

    def __init__(self):
        logger.info("Initializing SolarAnalyzer...")
        self._initialize_constants()
        self._initialize_cache()
        self.buildings_gdf = None
        self.spatial_index = None

    def _initialize_constants(self):
        self.NYC_LAT = 40.7128
        self.NYC_LON = -74.0060
        self.PANEL_EFFICIENCY = 0.20
        self.PERFORMANCE_RATIO = 0.75
        self.PANEL_DENSITY = 0.70
        self._solar_position = None

    def _initialize_cache(self):
        self._monthly_radiation = self._initialize_radiation_data()
        self._annual_radiation = self._calculate_annual_radiation()

    def _initialize_radiation_data(self):
        return {
            '01': 2.45, '02': 3.42, '03': 4.53, '04': 5.64,
            '05': 6.48, '06': 6.89, '07': 6.75, '08': 5.98,
            '09': 4.92, '10': 3.67, '11': 2.56, '12': 2.12
        }

    def _calculate_annual_radiation(self):
        monthly_sum = sum(self._monthly_radiation.values())
        return monthly_sum * 365 / 12

    def initialize_spatial_index(self, buildings_gdf):
        self.spatial_index = buildings_gdf.sindex
        return buildings_gdf

    def get_solar_position(self):
        if self._solar_position is None:
            times = pd.date_range('2020-06-21 12:00:00', periods=1, freq='H', tz='UTC')
            location = pvlib.location.Location(latitude=self.NYC_LAT, longitude=self.NYC_LON)
            self._solar_position = location.get_solarposition(times).iloc[0]
        return self._solar_position

def process_buildings_parallel(analyzer, buildings_chunk, buildings_gdf):
    """
    Process buildings using multiprocessing.
    """
    analyzer.buildings_gdf = buildings_gdf
    analyzer.initialize_spatial_index(buildings_gdf)

    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Using {num_processes} processes for parallel processing.")

    args_list = [(idx, row) for idx, row in buildings_chunk.iterrows()]

    solar_position = analyzer.get_solar_position()
    spatial_index = analyzer.spatial_index
    annual_radiation = analyzer._annual_radiation
    panel_density = analyzer.PANEL_DENSITY
    panel_efficiency = analyzer.PANEL_EFFICIENCY
    performance_ratio = analyzer.PERFORMANCE_RATIO

    with Pool(processes=num_processes, initializer=init_worker, initargs=(
        buildings_gdf, spatial_index, solar_position, annual_radiation,
        panel_density, panel_efficiency, performance_ratio
    )) as pool:
        results_list = pool.map(worker_process_building, args_list)

    return pd.DataFrame(results_list, index=buildings_chunk.index)

def analyze_solar_potential(candidate_buildings, full_buildings):
    """
    Analyze solar potential for candidate buildings.
    Here, candidate_buildings and full_buildings are the same dataset for simplicity.
    """
    try:
        analyzer = SolarAnalyzer()
        logger.info("Initialized SolarAnalyzer")

        # Ensure CRS is set - assuming EPSG:2263 for calculations
        if candidate_buildings.crs is None:
            logger.warning("Input data missing CRS, assuming EPSG:2263")
            candidate_buildings = candidate_buildings.set_crs(epsg=2263)
        if full_buildings.crs is None:
            full_buildings = full_buildings.set_crs(epsg=2263)

        candidate_projected = candidate_buildings.to_crs(epsg=2263)
        buildings_projected = full_buildings.to_crs(epsg=2263)

        analyzer.initialize_spatial_index(buildings_projected)

        results = process_buildings_parallel(
            analyzer,
            candidate_projected,
            buildings_projected
        )

        analyzed_buildings = candidate_projected.join(results)
        return analyzed_buildings.to_crs(candidate_buildings.crs)

    except Exception as e:
        logger.error(f"Error in solar potential analysis: {str(e)}")
        return None

if __name__ == "__main__":
    start_time = time.time()

    # Load the preprocessed sites
    input_file = Path('./output/preprocessed_sites.geojson')
    if not input_file.exists():
        logger.error("preprocessed_sites.geojson not found. Run preprocessing first.")
        exit(1)

    sites = gpd.read_file(input_file)
    logger.info(f"Loaded {len(sites)} sites from preprocessed_sites.geojson")

    # Ensure required columns
    if 'heightroof' not in sites.columns:
        logger.warning("heightroof not found, defaulting to 10 ft for demonstration.")
        sites['heightroof'] = 10.0
    else:
        # Convert heightroof to numeric and fill missing
        sites['heightroof'] = pd.to_numeric(sites['heightroof'], errors='coerce')
        missing_count = sites['heightroof'].isna().sum()
        if missing_count > 0:
            logger.warning(f"{missing_count} buildings had non-numeric heightroof values; assigning default of 10 ft.")
            sites['heightroof'].fillna(10.0, inplace=True)

    if 'Shape_Area' not in sites.columns:
        logger.info("Shape_Area not found, calculating from geometry...")
        if sites.crs is None:
            sites = sites.set_crs(epsg=2263)
        else:
            sites = sites.to_crs(epsg=2263)
        sites['Shape_Area'] = sites.geometry.area

    full_buildings = sites.copy()

    # Analyze solar potential
    logger.info("Starting solar potential analysis...")
    analyzed_buildings = analyze_solar_potential(sites, full_buildings)

    if analyzed_buildings is not None and not analyzed_buildings.empty:
        logger.info("Solar analysis completed successfully.")

        # Save as GeoJSON
        geojson_output = Path("./output/RHsites_solar.geojson")
        analyzed_buildings.to_file(geojson_output, driver='GeoJSON')
        logger.info(f"Results saved to {geojson_output}")

        # Save as Shapefile
        shapefile_output = Path("./output/shapefiles/RHsites_solar.shp")
        # Convert any non-standard data types to string to ensure shapefile compatibility
        for col in analyzed_buildings.select_dtypes(include=['object']).columns:
            analyzed_buildings[col] = analyzed_buildings[col].astype(str)
        # Limit field names to 10 characters for shapefile compatibility
        analyzed_buildings.columns = [col[:10] for col in analyzed_buildings.columns]
        analyzed_buildings.to_file(shapefile_output, driver='ESRI Shapefile')
        logger.info(f"Results saved to {shapefile_output}")
    else:
        logger.error("Solar analysis returned no results.")

    end_time = time.time()
    logger.info(f"Total solar analysis execution time: {end_time - start_time:.2f} seconds")