# process_results.py

from concurrent.futures import ProcessPoolExecutor
import time
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
import os
import geopandas as gpd

from solar_analysis import analyze_solar_potential, SolarAnalyzer

# Optional memory profiling
try:
    from memory_profiler import profile
except ImportError:
    def profile(func):
        return func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@profile
def initialize_parallel_processing(n_workers=None):
    """Initialize parallel processing with optimal workers"""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    logger.info(f"Initializing ProcessPoolExecutor with {n_workers} workers")
    return ProcessPoolExecutor(max_workers=n_workers)

@profile
def validate_input_data(candidate_buildings, buildings):
    """Validate input data and log summary statistics"""
    logger.info("\nInput Data Verification:")
    logger.info(f"Number of candidate buildings: {len(candidate_buildings)}")
    logger.info(f"Number of total buildings: {len(buildings)}")

    required_columns = ['heightroof', 'Shape_Area', 'geometry']
    for df, name in [(candidate_buildings, 'candidate_buildings'), (buildings, 'buildings')]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {name}: {missing_cols}")

    return True

@profile
def process_and_save_results(analyzed_buildings, output_dir='results'):
    """Process and save results with optimization"""
    try:
        # Calculate statistics
        positive_potential = analyzed_buildings[
            analyzed_buildings['solar_potential'] > 0.01
        ]

        # Save results
        save_path = Path(output_dir)
        save_path.mkdir(exist_ok=True)

        # Save to parquet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parquet_path = save_path / f'solar_analysis_results_{timestamp}.parquet'
        analyzed_buildings.to_parquet(
            parquet_path,
            engine='fastparquet'
        )
        logger.info(f"Results saved to: {parquet_path}")

        # Save debug information
        save_debug_info(analyzed_buildings, positive_potential, save_path, timestamp)

        return positive_potential

    except Exception as e:
        logger.error(f"Error in processing and saving results: {str(e)}")
        raise

@profile
def save_debug_info(analyzed_buildings, positive_potential, save_path, timestamp):
    """Save detailed debug information"""
    debug_path = save_path / f'solar_analysis_debug_{timestamp}.txt'

    with open(debug_path, 'w') as f:
        f.write("=== Solar Analysis Debug Output ===\n\n")

        # Basic statistics
        f.write("Basic Statistics:\n")
        f.write(f"Total buildings analyzed: {len(analyzed_buildings)}\n")
        f.write(f"Buildings with positive potential: {len(positive_potential)}\n\n")

        # Detailed statistics for positive potential buildings
        if len(positive_potential) > 0:
            f.write("Solar Potential Statistics:\n")
            f.write(f"Total potential: {positive_potential['solar_potential'].sum():,.0f} kWh/year\n")
            f.write(f"Average potential: {positive_potential['solar_potential'].mean():,.0f} kWh/year\n")
            f.write(f"Maximum potential: {positive_potential['solar_potential'].max():,.0f} kWh/year\n\n")

        # Column statistics
        f.write("Column Statistics:\n")
        for col in analyzed_buildings.select_dtypes(include=[np.number]).columns:
            f.write(f"\n{col}:\n")
            f.write(f"- Range: {analyzed_buildings[col].min():,.2f} to {analyzed_buildings[col].max():,.2f}\n")
            f.write(f"- Mean: {analyzed_buildings[col].mean():,.2f}\n")
            f.write(f"- Non-null count: {analyzed_buildings[col].count()}\n")

    logger.info(f"Debug information saved to: {debug_path}")

@profile
def print_analysis_summary(analyzed_buildings, positive_potential):
    """Print summary of analysis results"""
    print("\n=== Analysis Summary ===")
    print(f"Total buildings analyzed: {len(analyzed_buildings)}")
    print(f"Buildings with positive potential: {len(positive_potential)}")

    if len(positive_potential) > 0:
        print("\nSolar Potential Statistics:")
        print(f"Total potential: {positive_potential['solar_potential'].sum():,.0f} kWh/year")
        print(f"Average potential: {positive_potential['solar_potential'].mean():,.0f} kWh/year")
        print(f"Maximum potential: {positive_potential['solar_potential'].max():,.0f} kWh/year")

    print("\nFirst Building Details:")
    if len(analyzed_buildings) > 0:
        first_building = analyzed_buildings.iloc[0]
        print(f"- Area: {first_building['Shape_Area']:,.2f} sq ft")
        print(f"- Height: {first_building['heightroof']:,.2f} ft")
        print(f"- Solar potential: {first_building['solar_potential']:,.2f} kWh/year")
        print(f"- Shadow factor: {first_building['shadow_factor']:,.2f}")

@profile
def main(candidate_buildings, buildings, output_dir='results'):
    print("\n=== Starting Solar Analysis with Import Verification ===")

    try:
        # Initialize parallel processing
        executor = initialize_parallel_processing()

        # Create analyzer instance to verify class access
        analyzer = SolarAnalyzer()
        logger.info("\nSolarAnalyzer initialization successful")
        logger.info(f"Annual radiation calculation: {analyzer._annual_radiation:.2f} kWh/m²/year")

        # Validate input data
        validate_input_data(candidate_buildings, buildings)

        # Run analysis
        analyzed_buildings = analyze_solar_potential(candidate_buildings, buildings)

        if analyzed_buildings is None:
            raise ValueError("Analysis failed to produce valid results")

        # Process results
        positive_potential = process_and_save_results(analyzed_buildings, output_dir)

        # Print summary
        print_analysis_summary(analyzed_buildings, positive_potential)

        return analyzed_buildings, positive_potential

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
    finally:
        executor.shutdown()
        logger.info("Analysis complete. Parallel processing shutdown.")

def run_analysis(candidate_buildings, buildings, output_dir='results'):
    """
    Wrapper function to run the analysis and handle the output directory
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Run main analysis
    return main(candidate_buildings, buildings, output_dir)

# Entry point for script execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process solar analysis results')
    parser.add_argument('candidate_file', help='Path to candidate buildings file')
    parser.add_argument('buildings_file', help='Path to full buildings file')
    parser.add_argument('--output-dir', default='results', help='Output directory')

    args = parser.parse_args()

    # Load data
    candidate_buildings = gpd.read_file(args.candidate_file)
    buildings = gpd.read_file(args.buildings_file)

    # Run analysis
    run_analysis(candidate_buildings, buildings, args.output_dir)