import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import math
import contextlib
from multiprocessing import Manager

# Import the analysis and mapping functions
from solar_analysis import SolarAnalyzer, analyze_solar_potential
from solar_mapping import create_detailed_solar_map

def initialize_shared_resources():
    """
    Initialize shared resources for all processes
    """
    analyzer = SolarAnalyzer()
    # Pre-cache solar position and radiation data
    analyzer.get_solar_position()
    analyzer._monthly_radiation = analyzer.get_nasa_power_data()
    return analyzer

def process_chunk(data_tuple, shared_data=None):
    """
    Process a chunk of buildings in parallel with shared resources
    """
    try:
        buildings_chunk, full_buildings = data_tuple
        return analyze_solar_potential(buildings_chunk, full_buildings)
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return None

def run_analysis(candidate_buildings, full_buildings, output_dir='results', n_processes=None):
    """
    Run the complete solar analysis workflow with parallel processing
    """
    # Set up parallel processing
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Ensure at least 1 process

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        print("\nInitial Data Summary:")
        print(f"Number of candidate buildings: {len(candidate_buildings)}")
        print(f"Number of total buildings: {len(full_buildings)}")
        print(f"Candidate buildings CRS: {candidate_buildings.crs}")
        print(f"Full buildings CRS: {full_buildings.crs}")
        print("\nColumns present in candidate buildings:")
        for col in candidate_buildings.columns:
            print(f"- {col}: {candidate_buildings[col].dtype}")

        # Validate required columns
        required_columns = ['heightroof', 'Shape_Area', 'geometry']
        missing_columns = [col for col in required_columns if col not in candidate_buildings.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in candidate buildings: {missing_columns}")

        missing_columns = [col for col in required_columns if col not in full_buildings.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in full buildings: {missing_columns}")

        # Initialize shared resources
        print("\nInitializing shared resources...")
        shared_analyzer = initialize_shared_resources()

        # Prepare chunks for parallel processing
        chunk_size = math.ceil(len(candidate_buildings) / n_processes)
        chunks = [candidate_buildings.iloc[i:i + chunk_size] for i in range(0, len(candidate_buildings), chunk_size)]

        # Create tuples of (chunk, full_buildings) for processing
        process_tuples = [(chunk, full_buildings) for chunk in chunks]

        print(f"\nStarting parallel solar analysis with {n_processes} processes...")

        # Process chunks in parallel with shared resources
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(executor.map(partial(process_chunk, shared_data=shared_analyzer), process_tuples))

        # Filter out None results and combine
        results = [r for r in results if r is not None]
        if not results:
            raise ValueError("All chunks failed to process")

        analyzed_buildings = pd.concat(results, ignore_index=True)

        # Save results
        results_file = output_path / f'solar_analysis_results_{timestamp}.gpkg'
        analyzed_buildings.to_file(results_file, driver='GPKG')
        print(f"\nResults saved to: {results_file}")

        # Generate summary statistics
        print("\nAnalysis Summary:")
        valid_buildings = analyzed_buildings[
            (analyzed_buildings['solar_potential'].notna()) & 
            (analyzed_buildings['solar_potential'] > 0.01)
        ]

        print(f"Total buildings analyzed: {len(analyzed_buildings)}")
        print(f"Buildings with solar potential: {len(valid_buildings)}")
        print(f"Buildings with null solar potential: {analyzed_buildings['solar_potential'].isna().sum()}")
        print(f"Buildings with zero solar potential: {(analyzed_buildings['solar_potential'] == 0).sum()}")

        if len(valid_buildings) > 0:
            print("\nSolar Potential Statistics:")
            print(f"Total potential: {valid_buildings['solar_potential'].sum():,.0f} kWh/year")
            print(f"Average potential: {valid_buildings['solar_potential'].mean():,.0f} kWh/year")
            print(f"Median potential: {valid_buildings['solar_potential'].median():,.0f} kWh/year")
        else:
            print("\nWARNING: No buildings with valid solar potential found!")
            print("\nDiagnostic Information:")
            print("Input data validation:")
            print(f"Buildings with valid height: {(analyzed_buildings['heightroof'] > 0).sum()}")
            print(f"Buildings with valid area: {(analyzed_buildings['Shape_Area'] > 0).sum()}")
            print(f"Buildings with valid shadow factor: {(analyzed_buildings['shadow_factor'] > 0).sum()}")

            print("\nValue ranges:")
            numeric_columns = analyzed_buildings.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                print(f"{col}: {analyzed_buildings[col].min()} to {analyzed_buildings[col].max()}")

    except Exception as e:
        print(f"\nError in analysis: {str(e)}")
        raise
    finally:
        print("\nAnalysis complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run solar potential analysis on building footprints')
    parser.add_argument('candidate_file', help='Path to candidate buildings GeoJSON/Shapefile')
    parser.add_argument('buildings_file', help='Path to full buildings GeoJSON/Shapefile')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--processes', type=int, default=None, 
                      help='Number of processes to use (default: CPU count - 1)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with additional logging')
    args = parser.parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # Load both datasets
    candidate_buildings = gpd.read_file(args.candidate_file)
    full_buildings = gpd.read_file(args.buildings_file)

    run_analysis(candidate_buildings, full_buildings, args.output_dir, args.processes)