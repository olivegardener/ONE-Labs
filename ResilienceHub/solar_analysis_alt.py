"""
solar_analysis.py
Enhanced version with improved area calculations and solar potential analysis.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from numba import njit
import logging
import pvlib
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import warnings
from pathlib import Path
import time
import shapely.geometry
from math import cos, sin, radians, degrees, pi
from datetime import datetime, timedelta
import pytz

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_area(geometry):
    """
    Calculate area for EPSG:6539 geometry (US Survey Feet).

    Args:
        geometry: Shapely geometry in EPSG:6539

    Returns:
        tuple: (area_ft2, area_m2)
    """
    # In EPSG:6539, area is in square US survey feet
    area_ft2 = geometry.area
    # Convert to square meters
    area_m2 = area_ft2 * 0.092903

    return area_ft2, area_m2


def calculate_orientation_factor(azimuth):
    """
    Calculate roof orientation factor (0-1) based on azimuth.

    Args:
        azimuth: Roof azimuth in degrees (0=North, 90=East, etc.)

    Returns:
        float: Orientation factor between 0 and 1
    """
    # Optimal azimuth for Northern Hemisphere is 180° (South)
    optimal_azimuth = 180
    angle_diff = abs(azimuth - optimal_azimuth)

    # Convert difference to factor (1.0 for optimal, decreasing to 0.6 for worst)
    if angle_diff <= 90:
        return 1.0 - (0.4 * angle_diff / 90)
    else:
        return 0.6

def estimate_roof_pitch(building_type):
    """
    Estimate roof pitch based on building type.

    Args:
        building_type: String indicating building type

    Returns:
        float: Estimated roof pitch in degrees
    """
    pitch_estimates = {
        'residential': 30,
        'commercial': 15,
        'industrial': 10,
        'apartment': 20,
        'default': 10
    }
    return pitch_estimates.get(building_type.lower(), pitch_estimates['default'])

@njit
def calculate_shadow_length(height, sun_altitude):
    """
    Calculate shadow length using numba for performance.

    Args:
        height: Building height
        sun_altitude: Solar altitude angle in degrees

    Returns:
        float: Shadow length
    """
    if sun_altitude <= 0:
        return float('inf')
    return height / np.tan(np.radians(sun_altitude))

def get_sun_positions(latitude, longitude, date):
    """
    Calculate sun positions for a given date.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        date: Date to calculate positions for

    Returns:
        DataFrame: Solar positions throughout the day
    """
    location = pvlib.location.Location(latitude, longitude)
    times = pd.date_range(
        start=f"{date} 00:00:00",
        end=f"{date} 23:59:59",
        freq='15min',
        tz='UTC'
    )
    return location.get_solarposition(times)

class ShadowAnalyzer:
    """Class for analyzing building shadows."""

    def __init__(self, time_step_minutes=15):
        self.time_step = time_step_minutes

    def _calculate_shadow_geometry(self, building, azimuth, altitude):
        """
        Calculate shadow geometry for a building at given sun position.

        Args:
            building: Building geometry and attributes
            azimuth: Sun azimuth in degrees
            altitude: Sun altitude in degrees

        Returns:
            Shapely geometry of shadow or None if no shadow
        """
        try:
            if altitude <= 0:
                return None  # No shadow when sun is below horizon

            # Get building height and geometry
            height = float(building.get('heightroof', 0))
            if height <= 0:
                return None

            building_geometry = building.geometry

            # Calculate shadow length
            shadow_length = self._calculate_shadow_length(height, altitude)
            if shadow_length <= 0:
                return None

            # Calculate shadow direction (opposite to sun azimuth)
            shadow_angle = (azimuth + 180) % 360

            # Convert angle to radians
            angle_rad = np.radians(shadow_angle)

            # Calculate shadow offset
            dx = shadow_length * np.sin(angle_rad)
            dy = shadow_length * np.cos(angle_rad)

            # Create shadow polygon
            shadow = shapely.affinity.translate(
                building_geometry,
                xoff=-dx,
                yoff=-dy
            )

            # Union original building and shadow
            shadow_area = shadow.union(building_geometry)

            return shadow_area

        except Exception as e:
            logger.error(f"Error calculating shadow geometry: {str(e)}")
            return None

    def _calculate_shadow_length(self, height, altitude):
        """
        Calculate shadow length based on building height and sun altitude.

        Args:
            height: Building height
            altitude: Sun altitude in degrees

        Returns:
            float: Shadow length
        """
        if altitude <= 0:
            return float('inf')
        return height / np.tan(np.radians(altitude))

    def _calculate_shadow_intersection(self, shadow, nearby_buildings):
        """
        Calculate how much a shadow intersects with nearby buildings.

        Args:
            shadow: Shadow geometry
            nearby_buildings: GeoDataFrame of nearby buildings

        Returns:
            float: Intersection factor (0-1)
        """
        if shadow is None or nearby_buildings.empty:
            return 0.0

        try:
            # Calculate total shadow area
            shadow_area = shadow.area
            if shadow_area <= 0:
                return 0.0

            # Calculate intersection with each nearby building
            intersection_area = 0
            for _, building in nearby_buildings.iterrows():
                if building.geometry.intersects(shadow):
                    intersection = building.geometry.intersection(shadow)
                    intersection_area += intersection.area

            # Calculate intersection factor
            intersection_factor = min(1.0, intersection_area / shadow_area)
            return intersection_factor

        except Exception as e:
            logger.error(f"Error calculating shadow intersection: {str(e)}")
            return 0.0

    def calculate_daily_shadows(self, building, nearby_buildings, date, location):
        """
        Calculate shadow impacts throughout a day.

        Args:
            building: Target building geometry and attributes
            nearby_buildings: GeoDataFrame of nearby buildings
            date: Date to analyze
            location: (latitude, longitude) tuple

        Returns:
            float: Average shadow impact factor (0-1)
        """
        try:
            sun_positions = get_sun_positions(location[0], location[1], date)
            shadow_impacts = []

            for _, sun_pos in sun_positions.iterrows():
                if sun_pos['apparent_elevation'] > 0:
                    # Calculate shadow geometry
                    shadow = self._calculate_shadow_geometry(
                        building,
                        sun_pos['azimuth'],
                        sun_pos['apparent_elevation']
                    )

                    # Calculate shadow impact
                    if shadow is not None:
                        impact = self._calculate_shadow_intersection(shadow, nearby_buildings)
                        shadow_impacts.append(1.0 - impact)

            # Calculate average impact
            if shadow_impacts:
                return np.mean(shadow_impacts)
            return 1.0  # No shadow impact

        except Exception as e:
            logger.error(f"Error calculating daily shadows: {str(e)}")
            return 1.0  # Return no impact in case of error

class SolarAnalyzer:
    """Enhanced class for analyzing solar potential of buildings."""

    def __init__(self):
        logger.info("Initializing enhanced SolarAnalyzer...")
        self._initialize_constants()
        self._initialize_cache()
        self.shadow_analyzer = ShadowAnalyzer()

    def _initialize_constants(self):
        """Initialize constants with more realistic values."""
        self.NYC_LAT = 40.7128
        self.NYC_LON = -74.0060

        # Enhanced panel characteristics
        self.PANEL_EFFICIENCY = 0.20  # 20% efficient panels
        self.PERFORMANCE_RATIO = 0.75  # System losses
        self.BASE_PANEL_DENSITY = 0.70  # Base density before adjustments

        # Temperature coefficients
        self.TEMP_COEFF = -0.0040  # Typical temperature coefficient (%/°C)
        self.NOMINAL_OPERATING_TEMP = 25  # Celsius

        # Maintenance factors
        self.SOILING_FACTOR = 0.97  # 3% loss from dirt/dust
        self.AGING_FACTOR = 0.99  # 1% annual degradation

        self._solar_position = None

    def _initialize_cache(self):
        """Initialize radiation data with monthly variations."""
        self._monthly_radiation = self._initialize_radiation_data()
        self._annual_radiation = self._calculate_annual_radiation()
        self._temperature_data = self._initialize_temperature_data()

    def _initialize_radiation_data(self):
        """Enhanced monthly radiation data with more precise values."""
        return {
            '01': 2.45, '02': 3.42, '03': 4.53, '04': 5.64,
            '05': 6.48, '06': 6.89, '07': 6.75, '08': 5.98,
            '09': 4.92, '10': 3.67, '11': 2.56, '12': 2.12
        }

    def _calculate_annual_radiation(self):
        """
        Calculate annual radiation from monthly values.

        Returns:
            float: Annual radiation in kWh/m²/year
        """
        # Calculate daily average for each month
        monthly_totals = []
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for month, days in zip(self._monthly_radiation.values(), days_in_month):
            monthly_totals.append(month * days)

        # Return annual total
        return sum(monthly_totals)

    def _initialize_temperature_data(self):
        """Initialize average monthly temperatures for NYC."""
        return {
            '01': 0.5, '02': 2.1, '03': 6.3, '04': 12.1,
            '05': 17.7, '06': 22.9, '07': 25.4, '08': 24.9,
            '09': 20.8, '10': 14.9, '11': 9.4, '12': 3.9
        }

    def calculate_temperature_efficiency(self, month):
        """Calculate temperature-adjusted efficiency."""
        avg_temp = self._temperature_data[month]
        temp_diff = avg_temp - self.NOMINAL_OPERATING_TEMP
        efficiency_adjustment = 1 + (self.TEMP_COEFF * temp_diff)
        return max(0.7, min(1.0, efficiency_adjustment))

    def calculate_adjusted_panel_density(self, roof_pitch, building_type):
        """
        Calculate adjusted panel density based on roof characteristics.

        Args:
            roof_pitch: Roof pitch in degrees
            building_type: Type of building

        Returns:
            float: Adjusted panel density
        """
        # Base adjustment for pitch
        pitch_factor = cos(radians(roof_pitch))

        # Building type specific adjustments
        type_factors = {
            'residential': 0.95,
            'commercial': 0.85,
            'industrial': 0.80,
            'apartment': 0.90
        }
        type_factor = type_factors.get(building_type.lower(), 0.85)

        # Calculate final density
        adjusted_density = self.BASE_PANEL_DENSITY * pitch_factor * type_factor

        # Account for spacing and mounting requirements
        spacing_factor = 0.95  # 5% loss for panel spacing
        mounting_factor = 0.98  # 2% loss for mounting hardware

        return adjusted_density * spacing_factor * mounting_factor

    def calculate_solar_potential(self, building_data, nearby_buildings):
        """
        Calculate comprehensive solar potential for a building.

        Args:
            building_data: GeoSeries containing building information
            nearby_buildings: GeoDataFrame of nearby buildings

        Returns:
            dict: Detailed solar potential calculations
        """
        try:
            # Get basic building characteristics
            geometry = building_data.geometry
            building_type = building_data.get('building_type', 'default')
            roof_pitch = building_data.get('roof_pitch', 
                                         estimate_roof_pitch(building_type))

            # Calculate areas - use the known CRS EPSG:6539
            # Since we know geometry is in EPSG:6539, area will be in square feet
            area_ft2 = geometry.area
            area_m2 = area_ft2 * 0.092903  # Convert sq ft to sq m

            # Calculate orientation factor
            orientation = building_data.get('orientation', 180)  # default to south
            orientation_factor = calculate_orientation_factor(orientation)

            # Calculate shadow factor
            shadow_factor = self.shadow_analyzer.calculate_daily_shadows(
                building_data,
                nearby_buildings,
                datetime.now().date(),
                (self.NYC_LAT, self.NYC_LON)
            )

            # Calculate adjusted panel density
            adjusted_density = self.calculate_adjusted_panel_density(
                roof_pitch,
                building_type
            )

            # Calculate monthly production
            monthly_production = {}
            annual_production = 0

            for month in self._monthly_radiation.keys():
                # Temperature adjusted efficiency
                temp_efficiency = self.calculate_temperature_efficiency(month)

                # Monthly radiation with all factors
                monthly_radiation = (
                    self._monthly_radiation[month] *
                    area_m2 *
                    adjusted_density *
                    self.PANEL_EFFICIENCY *
                    temp_efficiency *
                    self.PERFORMANCE_RATIO *
                    shadow_factor *
                    orientation_factor *
                    self.SOILING_FACTOR *
                    self.AGING_FACTOR
                )

                monthly_production[month] = monthly_radiation
                annual_production += monthly_radiation

            # Calculate peak power capacity
            peak_power = (
                area_m2 *
                adjusted_density *
                self.PANEL_EFFICIENCY *
                1000  # Convert to Watts
            )

            return {
                'solar_potential': annual_production,
                'peak_power': peak_power,
                'effective_area': area_m2 * adjusted_density,
                'shadow_factor': shadow_factor,
                'orientation_factor': orientation_factor,
                'monthly_production': monthly_production,
                'area_ft2': area_ft2,
                'area_m2': area_m2,
                'adjusted_density': adjusted_density
            }

        except Exception as e:
            logger.error(f"Error calculating solar potential: {str(e)}")
            return None

def process_building_parallel(args):
    """
    Enhanced parallel processing function for individual buildings.

    Args:
        args: Tuple containing (idx, building_row, context_data)

    Returns:
        dict: Processed building data with solar calculations
    """
    idx, building, context = args
    try:
        analyzer = context['analyzer']
        nearby_buildings = context['buildings_gdf']

        # Calculate solar potential
        results = analyzer.calculate_solar_potential(building, nearby_buildings)

        if results is None:
            return create_default_results(idx)

        # Add metadata
        results.update({
            'building_id': idx,
            'calculation_timestamp': datetime.now().isoformat(),
            'calculation_version': '2.0'
        })

        return results

    except Exception as e:
        logger.error(f"Error processing building {idx}: {str(e)}")
        return create_default_results(idx)

def create_default_results(idx):
    """Create default results for failed calculations."""
    return {
        'building_id': idx,
        'solar_potential': 0.0,
        'peak_power': 0.0,
        'effective_area': 0.0,
        'shadow_factor': 0.0,
        'orientation_factor': 0.0,
        'monthly_production': {str(m).zfill(2): 0.0 for m in range(1, 13)},
        'area_ft2': 0.0,
        'area_m2': 0.0,
        'adjusted_density': 0.0,
        'calculation_status': 'failed'
    }

def analyze_solar_potential(candidate_buildings, full_buildings):
    """
    Analyze solar potential for candidate buildings.

    Args:
        candidate_buildings (GeoDataFrame): Buildings to analyze
        full_buildings (GeoDataFrame): Complete building dataset for context

    Returns:
        GeoDataFrame: Buildings with solar potential calculations
    """
    try:
        analyzer = SolarAnalyzer()
        logger.info("Initialized SolarAnalyzer")

        # Create a deep copy to preserve all data including geometries
        analyzed_buildings = candidate_buildings.copy(deep=True)

        # Debug geometry check
        logger.info(f"Initial geometry count: {len(analyzed_buildings[analyzed_buildings.geometry.is_valid])}")

        # Ensure CRS is set and transform to EPSG:6539 if needed
        if analyzed_buildings.crs is None:
            logger.warning("Input CRS is None, assuming EPSG:4326")
            analyzed_buildings = analyzed_buildings.set_crs('EPSG:4326')
            full_buildings = full_buildings.set_crs('EPSG:4326')

        # Project to NAD83(2011) / New York Long Island (ftUS) for accurate calculations
        if analyzed_buildings.crs != 'EPSG:6539':
            analyzed_buildings = analyzed_buildings.to_crs('EPSG:6539')
            full_buildings = full_buildings.to_crs('EPSG:6539')

        # Debug geometry check after projection
        logger.info(f"Geometry count after projection: {len(analyzed_buildings[analyzed_buildings.geometry.is_valid])}")

        # Initialize spatial index for shadow calculations
        analyzer.initialize_spatial_index(full_buildings)

        # Calculate areas
        analyzed_buildings['area_ft2'] = analyzed_buildings.geometry.area
        analyzed_buildings['area_m2'] = analyzed_buildings['area_ft2'] * 0.092903

        # Get solar position and radiation data
        solar_position = analyzer.get_solar_position()
        annual_radiation = analyzer._annual_radiation

        # Create list to store results while preserving geometries
        solar_results = []

        for idx, building in analyzed_buildings.iterrows():
            try:
                if not building.geometry.is_valid:
                    logger.warning(f"Invalid geometry found for building {idx}")
                    continue

                # Calculate shadow factor
                shadow_factor = calculate_shadow_impact_vectorized(
                    building, 
                    full_buildings, 
                    solar_position, 
                    analyzer.spatial_index
                )

                # Calculate effective area (accounting for panel density)
                effective_area = building['area_m2'] * analyzer.PANEL_DENSITY

                # Calculate peak power potential
                peak_power = (
                    effective_area * 
                    analyzer.PANEL_EFFICIENCY
                )

                # Calculate total solar potential
                solar_potential = (
                    annual_radiation *
                    effective_area *
                    analyzer.PANEL_EFFICIENCY *
                    analyzer.PERFORMANCE_RATIO *
                    shadow_factor
                )

                result_dict = {
                    'solar_potential': float(solar_potential),
                    'effective_area': float(effective_area),
                    'peak_power': float(peak_power),
                    'shadow_factor': float(shadow_factor),
                    'annual_radiation': float(annual_radiation),
                    'performance_ratio': float(analyzer.PERFORMANCE_RATIO),
                    'area_ft2': float(building['area_ft2']),
                    'area_m2': float(building['area_m2']),
                    'geometry': building.geometry  # Preserve the geometry
                }

                solar_results.append(result_dict)

            except Exception as e:
                logger.error(f"Error processing building {idx}: {str(e)}")
                # Add null results while preserving geometry
                solar_results.append({
                    'solar_potential': 0.0,
                    'effective_area': 0.0,
                    'peak_power': 0.0,
                    'shadow_factor': 0.0,
                    'annual_radiation': 0.0,
                    'performance_ratio': 0.0,
                    'area_ft2': 0.0,
                    'area_m2': 0.0,
                    'geometry': building.geometry  # Preserve the geometry
                })

        # Create new GeoDataFrame from results
        results_gdf = gpd.GeoDataFrame(
            solar_results,
            crs=analyzed_buildings.crs,
            geometry='geometry'
        )

        # Debug geometry check
        logger.info(f"Final geometry count: {len(results_gdf[results_gdf.geometry.is_valid])}")

        # Project back to original CRS if needed
        if candidate_buildings.crs != 'EPSG:6539':
            results_gdf = results_gdf.to_crs(candidate_buildings.crs)

        # Copy over any additional columns from original data
        for col in candidate_buildings.columns:
            if col not in results_gdf.columns and col != 'geometry':
                results_gdf[col] = candidate_buildings[col]

        logger.info(f"Completed solar analysis for {len(results_gdf)} buildings")
        return results_gdf

    except Exception as e:
        logger.error(f"Error in solar potential analysis: {str(e)}")
        return None



def process_buildings_parallel(sites):
    """
    Process buildings in parallel while preserving geometries.
    """
    try:
        num_processes = max(1, cpu_count() - 1)
        logger.info(f"Processing {len(sites)} buildings using {num_processes} processes")

        # Store original geometries
        original_geometries = sites.geometry.copy()

        # Process solar potential
        analyzer = SolarAnalyzer()
        logger.info("Initializing enhanced SolarAnalyzer...")

        # Process buildings and calculate solar potential
        results = analyze_solar_potential(sites, sites)

        # Ensure results is a GeoDataFrame
        if not isinstance(results, gpd.GeoDataFrame):
            results = gpd.GeoDataFrame(results)

        # Restore original geometries
        results['geometry'] = original_geometries
        results = results.set_crs(sites.crs)

        return results

    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        return None


def validate_and_clean_data(gdf):
    """
    Validate and clean input data.

    Args:
        gdf: Input GeoDataFrame

    Returns:
        GeoDataFrame: Cleaned and validated data
    """
    if gdf is None or len(gdf) == 0:
        raise ValueError("Empty or invalid input data")

    # Ensure required columns
    required_columns = ['geometry', 'heightroof']
    missing_columns = [col for col in required_columns if col not in gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert heightroof to numeric
    gdf['heightroof'] = pd.to_numeric(gdf['heightroof'], errors='coerce')
    gdf['heightroof'].fillna(0, inplace=True)

    # Validate geometries
    gdf = gdf[gdf.geometry.is_valid]
    gdf = gdf[~gdf.geometry.is_empty]

    return gdf


def calculate_summary_statistics(gdf):
    """Calculate summary statistics for the analysis."""
    try:
        stats_dict = {
            'Metric': [
                'Total Buildings',
                'Total Solar Potential (kWh/year)',
                'Average Solar Potential (kWh/year)',
                'Total Peak Power (kW)',
                'Total Effective Area (m²)',
                'Average Shadow Factor',
                'Analysis Timestamp'
            ],
            'Value': [
                len(gdf),
                gdf['solar_potential'].sum(),
                gdf['solar_potential'].mean(),
                gdf['peak_power'].sum(),
                gdf['effective_area'].sum(),
                gdf['shadow_factor'].mean(),
                datetime.now().isoformat()
            ]
        }

        # Create DataFrame with proper indexing
        return pd.DataFrame(stats_dict)

    except Exception as e:
        logger.error(f"Error calculating summary statistics: {str(e)}")
        # Return minimal statistics if error occurs
        return pd.DataFrame({
            'Metric': ['Error', 'Timestamp'],
            'Value': [str(e), datetime.now().isoformat()]
        })



def validate_and_repair_geometries(gdf):
    """
    Validate and attempt to repair invalid geometries in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame
    Returns:
        GeoDataFrame: GeoDataFrame with validated/repaired geometries
    """
    if gdf is None or len(gdf) == 0:
        return None

    try:
        logger.info(f"Validating and repairing geometries for {len(gdf)} features...")

        # Make a copy to avoid modifying the original
        validated_gdf = gdf.copy()

        # Count initial invalid geometries
        invalid_count = sum(~validated_gdf.geometry.is_valid)
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid geometries")

        # Function to repair individual geometry
        def repair_geometry(geom):
            if geom is None:
                return None
            try:
                if not geom.is_valid:
                    # Try buffer(0) first
                    repaired = geom.buffer(0)
                    if not repaired.is_valid:
                        # If buffer(0) fails, try make_valid()
                        repaired = shapely.make_valid(geom)
                    return repaired
                return geom
            except Exception as e:
                logger.error(f"Error repairing geometry: {str(e)}")
                return None

        # Repair geometries
        validated_gdf.geometry = validated_gdf.geometry.apply(repair_geometry)

        # Remove any null geometries
        validated_gdf = validated_gdf[validated_gdf.geometry.notna()]

        # Final validity check
        final_invalid_count = sum(~validated_gdf.geometry.is_valid)

        logger.info(f"Geometry validation complete:")
        logger.info(f"- Initial invalid geometries: {invalid_count}")
        logger.info(f"- Remaining invalid geometries: {final_invalid_count}")
        logger.info(f"- Valid features remaining: {len(validated_gdf)}")

        return validated_gdf

    except Exception as e:
        logger.error(f"Error in geometry validation: {str(e)}")
        return gdf

def create_unique_truncated_columns(columns, max_length=10):
    """
    Create unique truncated column names.

    Args:
        columns: List of column names
        max_length: Maximum length for truncated names

    Returns:
        dict: Mapping of original to truncated names
    """
    truncated_names = {}
    seen_names = set()

    for col in columns:
        truncated = col[:max_length]
        base_name = truncated
        counter = 1

        # If truncated name already exists, add number suffix
        while truncated in seen_names:
            # Calculate space needed for counter
            counter_str = str(counter)
            name_length = max_length - len(counter_str)
            truncated = f"{base_name[:name_length]}{counter}"
            counter += 1

        seen_names.add(truncated)
        truncated_names[col] = truncated

    return truncated_names

def create_centroids_within_buildings(buildings_gdf):
    """
    Create a single point within each building polygon.
    """
    if buildings_gdf is None or len(buildings_gdf) == 0:
        logger.error("Input GeoDataFrame is empty or None")
        return None

    try:
        # Validate and repair geometries first
        validated_gdf = validate_and_repair_geometries(buildings_gdf)
        if validated_gdf is None or len(validated_gdf) == 0:
            logger.error("No valid geometries after validation")
            return None

        logger.info(f"Creating centroids for {len(validated_gdf)} buildings...")
        points_data = []

        # Process each building
        for idx, building in validated_gdf.iterrows():
            try:
                if building.geometry is None or building.geometry.is_empty:
                    continue

                point = building.geometry.centroid
                if not building.geometry.contains(point):
                    point = building.geometry.representative_point()

                # Create point data dictionary
                point_data = {'geometry': point}

                # Add all columns except geometry
                for col in validated_gdf.columns:
                    if col != 'geometry':
                        point_data[col] = building[col]

                points_data.append(point_data)

            except Exception as e:
                logger.error(f"Error processing building {idx}: {str(e)}")
                continue

        if not points_data:
            logger.error("No valid points were created")
            return None

        # Create GeoDataFrame from points
        points_gdf = gpd.GeoDataFrame(
            points_data,
            crs=validated_gdf.crs
        )

        # Add point ID field
        points_gdf['point_id'] = range(1, len(points_gdf) + 1)

        # Ensure all numeric columns are float type
        numeric_columns = points_gdf.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            points_gdf[col] = points_gdf[col].astype(float)

        # Create unique truncated column names
        column_mapping = create_unique_truncated_columns(points_gdf.columns)
        points_gdf.columns = [column_mapping[col] for col in points_gdf.columns]

        logger.info(f"Successfully created {len(points_gdf)} centroids")
        return points_gdf

    except Exception as e:
        logger.error(f"Error creating centroids within buildings: {str(e)}")
        return None

def save_results(gdf, output_dir):
    """Save results in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save GeoJSON (full column names)
        if gdf is not None and len(gdf) > 0:
            gdf.to_file(
                output_dir / "sites_solar.geojson",
                driver='GeoJSON',
                encoding='utf-8'
            )
            logger.info(f"Saved {len(gdf)} records to sites_solar.geojson")

        # Create and save centroids
        centroids_gdf = create_centroids_within_buildings(gdf)
        if centroids_gdf is not None and len(centroids_gdf) > 0:
            # Save GeoJSON version (can handle longer column names)
            centroids_gdf_geojson = centroids_gdf.copy()
            centroids_gdf_geojson.to_file(
                output_dir / "sites_solar_points.geojson",
                driver='GeoJSON',
                encoding='utf-8'
            )
            logger.info(f"Saved {len(centroids_gdf)} points to sites_solar_points.geojson")

            # Save shapefile version (with truncated column names)
            shapefile_dir = output_dir / "shapefiles"
            shapefile_dir.mkdir(exist_ok=True)

            centroids_gdf.to_file(
                shapefile_dir / "sites_solar_points.shp",
                encoding='utf-8'
            )
            logger.info(f"Saved points shapefile to {shapefile_dir}")

        # Save summary statistics
        summary_stats = calculate_summary_statistics(gdf)
        if summary_stats is not None:
            summary_stats.to_csv(
                output_dir / "solar_analysis_summary.csv",
                index=False
            )
            logger.info("Saved summary statistics to solar_analysis_summary.csv")

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise



if __name__ == "__main__":
    try:
        start_time = time.time()

        # Load input data
        input_file = Path('./output/preprocessed_sites_solar.geojson')
        if not input_file.exists():
            raise FileNotFoundError("Input file not found. Run preprocessing first.")

        # Load and validate data
        sites = gpd.read_file(input_file)
        logger.info(f"Loaded {len(sites)} sites from {input_file}")

        # Validate and clean data
        sites = validate_and_clean_data(sites)
        logger.info(f"Validated data: {len(sites)} valid sites remaining")

        # Ensure proper projection
        if sites.crs is None or not sites.crs.is_projected:
            sites = sites.to_crs('EPSG:6539')
            logger.info("Projected data to EPSG:6539 for accurate calculations")

        # Sample data more safely
        sample_size = max(1, int(len(sites) * 1))  # Calculate 1% but ensure at least 1 record

        # Get random indices for sampling
        sample_indices = np.random.choice(
            sites.index, 
            size=sample_size, 
            replace=False
        )

        # Create sample using loc to maintain index and geometry validity
        sites_sample = sites.loc[sample_indices].copy()

        # Validate geometries after sampling
        sites_sample = validate_and_repair_geometries(sites_sample)

        if sites_sample is None or len(sites_sample) == 0:
            raise ValueError("No valid geometries after sampling")

        logger.info(f"Sampled {len(sites_sample)} sites for processing")

        # Process buildings
        results = process_buildings_parallel(sites_sample)

        # Save results
        save_results(results, './output')

        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        logger.info("Solar analysis completed successfully")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise
