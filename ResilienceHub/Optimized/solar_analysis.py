import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
import warnings
import json
from pathlib import Path
import requests
from functools import lru_cache
import time  # Add this import
from time import sleep
import os
from datetime import datetime, timedelta
from multiprocessing import Lock, Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import geopandas as gpd
from shapely.prepared import prep
import dask.dataframe as dd
from tqdm import tqdm
import numba
from numba import jit
import logging

# Optional memory profiling
try:
    from memory_profiler import profile
except ImportError:
    def profile(func):
        return func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Configuration constants
CHUNK_SIZE = 1000  # Adjust based on memory usage
N_PROCESSES = max(1, cpu_count() - 1)  # Leave one core free
SPATIAL_INDEX = True
__version__ = '1.0.0'

# Enhanced numba optimizations
@numba.jit(nopython=True, parallel=True)
def calculate_shadow_impact_batch(heights, distances, angles, alts):
    results = np.zeros(len(heights))
    for i in numba.prange(len(heights)):
        results[i] = calculate_shadow_impact_numba(
            heights[i], alts[i], distances[i], angles[i]
        )
    return results

@numba.jit(nopython=True)
def calculate_shadow_impact_numba(height, alt, distance, angle):
    """Calculate the shadow impact of a building"""
    if alt <= 0:
        # Sun is below horizon, full shadow
        return 1.0
    shadow_length = height / np.tan(np.radians(alt))
    if distance <= shadow_length:
        # Building is within shadow length
        shadow_factor = 1.0 - (distance / shadow_length)
        return shadow_factor
    else:
        # No shadow impact
        return 0.0

class SolarAnalyzer:
    def __init__(self):
        logger.info("Initializing SolarAnalyzer...")
        self._initialize_constants()
        self._initialize_cache()

    def _initialize_constants(self):
        self.NYC_LAT = 40.7128
        self.NYC_LON = -74.0060
        self.PANEL_EFFICIENCY = 0.20
        self.PERFORMANCE_RATIO = 0.75
        self.PANEL_DENSITY = 0.70
        self._solar_position = None
        self._spatial_index = None

    def _initialize_cache(self):
        self._cache = {}
        self._monthly_radiation = self._initialize_radiation_data()
        self._annual_radiation = self._calculate_annual_radiation()

    def _initialize_radiation_data(self):
        """Initialize default radiation data"""
        return {
            '01': 2.45, '02': 3.42, '03': 4.53, '04': 5.64,
            '05': 6.48, '06': 6.89, '07': 6.75, '08': 5.98,
            '09': 4.92, '10': 3.67, '11': 2.56, '12': 2.12
        }

    def _calculate_annual_radiation(self):
        """Calculate annual radiation from monthly data"""
        monthly_sum = sum(self._monthly_radiation.values())
        return monthly_sum * 365 / 12

    def _calculate_angles_batch(self, nearby_buildings, reference_building):
        """Calculate angles between buildings in batch"""
        ref_centroid = reference_building.geometry.centroid
        nearby_centroids = nearby_buildings.geometry.centroid

        dx = nearby_centroids.x.values - ref_centroid.x
        dy = nearby_centroids.y.values - ref_centroid.y

        return np.degrees(np.arctan2(dy, dx))

    def _calculate_geometric_parameters(self, building_id, solar_position):
        """Calculate geometric parameters for a building"""
        # Implementation depends on your specific requirements
        pass

    def _calculate_partition_shadow(self, buildings_partition, reference_building, solar_pos):
        """Calculate shadow impact for a partition of buildings"""
        # Implementation for dask partitioning
        pass

    def _process_single_building(self, building, buildings_gdf):
        """Process a single building"""
        area_m2 = building['Shape_Area'] * 0.092903  # Convert from square feet to square meters
        shadow_factor = self.calculate_shadow_impact_vectorized(
            building, buildings_gdf, self.get_solar_position()
        )

        return {
            'solar_potential': float(
                self._annual_radiation *
                area_m2 *
                self.PANEL_DENSITY *
                self.PANEL_EFFICIENCY *
                self.PERFORMANCE_RATIO *
                shadow_factor
            ),
            'shadow_factor': float(shadow_factor),
            'effective_area': float(area_m2 * self.PANEL_DENSITY),
            'peak_power': float(area_m2 * self.PANEL_DENSITY * self.PANEL_EFFICIENCY),
            'annual_radiation': float(self._annual_radiation),
            'performance_ratio': float(self.PERFORMANCE_RATIO)
        }

    @staticmethod
    def optimize_memory_usage(gdf):
        """Optimize GeoDataFrame memory usage"""
        for col in gdf.select_dtypes(include=['object']):
            gdf[col] = gdf[col].astype('category')

        for col in gdf.select_dtypes(include=['float']):
            gdf[col] = pd.to_numeric(gdf[col], downcast='float')

        return gdf

    @lru_cache(maxsize=1024)
    def cached_geometric_calculations(self, building_id, solar_position):
        """Cache geometric calculations for frequently accessed buildings"""
        if building_id in self._cache:
            return self._cache[building_id]

        result = self._calculate_geometric_parameters(building_id, solar_position)
        self._cache[building_id] = result
        return result

    def calculate_shadow_impact_vectorized(self, building_row, buildings_gdf, solar_pos):
        """Enhanced vectorized shadow impact calculation"""
        try:
            # Optimize input data
            buildings_gdf = self.optimize_memory_usage(buildings_gdf)

            # Use dask for large calculations
            if len(buildings_gdf) > 10000:
                return self._calculate_shadow_impact_dask(
                    building_row, buildings_gdf, solar_pos
                )

            # Original vectorized calculation with batch processing
            MAX_SHADOW_LENGTH = 3 * building_row['heightroof']
            buffer_geom = building_row.geometry.buffer(MAX_SHADOW_LENGTH)

            # Use spatial index efficiently
            if SPATIAL_INDEX and self._spatial_index is not None:
                possible_matches_idx = list(self._spatial_index.intersection(buffer_geom.bounds))
                nearby = buildings_gdf.iloc[possible_matches_idx]
            else:
                nearby = buildings_gdf[buildings_gdf.geometry.intersects(buffer_geom)]

            # Remove the building itself from the nearby buildings
            nearby = nearby[nearby.index != building_row.name]

            # Batch process shadow calculations
            heights = nearby['heightroof'].values
            distances = nearby.geometry.distance(building_row.geometry).values
            angles = self._calculate_angles_batch(nearby, building_row)
            alts = np.full_like(heights, solar_pos['apparent_elevation'])

            if len(heights) == 0:
                return 1.0  # No nearby buildings to cause shadows

            shadow_factors = calculate_shadow_impact_batch(
                heights, distances, angles, alts
            )

            # Calculate overall shadow factor
            overall_shadow_factor = 1.0 - np.mean(shadow_factors)
            return overall_shadow_factor

        except Exception as e:
            logger.error(f"Error in shadow calculation: {str(e)}")
            return 1.0

    def _calculate_shadow_impact_dask(self, building_row, buildings_gdf, solar_pos):
        """Dask implementation for large datasets"""
        ddf = dd.from_pandas(buildings_gdf, npartitions=N_PROCESSES)
        result = ddf.map_partitions(
            self._calculate_partition_shadow,
            building_row=building_row,
            solar_pos=solar_pos,
            meta=('shadow_factor', 'float64')
        ).compute()
        return float(result.mean())

    def process_buildings_vectorized(self, buildings_chunk, buildings_gdf):
        """Memory-optimized vectorized processing"""
        try:
            # Optimize input data
            buildings_chunk = self.optimize_memory_usage(buildings_chunk)
            buildings_gdf = self.optimize_memory_usage(buildings_gdf)

            results = []
            for idx, building in buildings_chunk.iterrows():
                result = self._process_single_building(building, buildings_gdf)
                results.append(result)

            return pd.DataFrame(results, index=buildings_chunk.index)

        except Exception as e:
            logger.error(f"Error in vectorized processing: {str(e)}")
            return pd.DataFrame()

    def initialize_spatial_index(self, buildings_gdf):
        """Initialize spatial index for buildings"""
        self._spatial_index = buildings_gdf.sindex
        return buildings_gdf

    def get_solar_position(self):
        """Calculate solar position"""
        if self._solar_position is None:
            times = pd.date_range('2020-06-21 12:00:00', periods=1, freq='H', tz='UTC')  # Summer solstice at noon
            location = Location(latitude=self.NYC_LAT, longitude=self.NYC_LON)
            self._solar_position = location.get_solarposition(times).iloc[0]
        return self._solar_position

# Main function outside the class
@profile
def analyze_solar_potential(candidate_buildings, full_buildings):
    """
    Main function to analyze solar potential for candidate buildings
    """
    try:
        # Initialize analyzer
        analyzer = SolarAnalyzer()
        logger.info("Initialized SolarAnalyzer")

        # Validate CRS
        if candidate_buildings.crs is None or full_buildings.crs is None:
            logger.warning("Input data missing CRS, assuming EPSG:4326")
            candidate_buildings.set_crs(epsg=4326, inplace=True)
            full_buildings.set_crs(epsg=4326, inplace=True)

        # Project to NYC State Plane Coordinate System (EPSG:2263)
        candidate_projected = candidate_buildings.to_crs(epsg=2263)
        buildings_projected = full_buildings.to_crs(epsg=2263)

        # Initialize spatial index
        analyzer.initialize_spatial_index(buildings_projected)

        # Process buildings
        results = analyzer.process_buildings_vectorized(
            candidate_projected, 
            buildings_projected
        )

        # Merge results back to original GeoDataFrame
        analyzed_buildings = candidate_projected.join(results)

        return analyzed_buildings

    except Exception as e:
        logger.error(f"Error in solar potential analysis: {str(e)}")
        return None

# Export the main functions
__all__ = ['SolarAnalyzer', 'analyze_solar_potential']