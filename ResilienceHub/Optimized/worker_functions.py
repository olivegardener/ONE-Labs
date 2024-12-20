# worker_functions.py

import numpy as np
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global variables for worker processes
global_analyzer = None
global_buildings_gdf = None

def init_worker(analyzer, buildings_gdf):
    """Initialize global variables in worker processes."""
    global global_analyzer
    global global_buildings_gdf
    global_analyzer = analyzer
    global_buildings_gdf = buildings_gdf

def worker_process_building(args):
    """Worker function to process a single building using global variables."""
    idx, building = args
    area_m2 = building['Shape_Area'] * 0.092903  # Convert from square feet to square meters

    # Access global variables
    analyzer = global_analyzer
    buildings_gdf = global_buildings_gdf

    # Calculate the shadow factor
    shadow_factor = analyzer.calculate_shadow_impact_vectorized(
        building, buildings_gdf, analyzer.get_solar_position()
    )

    solar_potential = (
        analyzer._annual_radiation *
        area_m2 *
        analyzer.PANEL_DENSITY *
        analyzer.PANEL_EFFICIENCY *
        analyzer.PERFORMANCE_RATIO *
        shadow_factor
    )

    return {
        'solar_potential': float(solar_potential),
        'effective_area': float(area_m2 * analyzer.PANEL_DENSITY),
        'peak_power': float(area_m2 * analyzer.PANEL_DENSITY * analyzer.PANEL_EFFICIENCY),
        'shadow_factor': float(shadow_factor),
        'annual_radiation': float(analyzer._annual_radiation),
        'performance_ratio': float(analyzer.PERFORMANCE_RATIO)
    }