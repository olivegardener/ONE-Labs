# run_solar_analysis.py

import time
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
import logging
import dask.dataframe as dd
import dask.distributed as distributed

from solar_analysis import SolarAnalyzer

# Optional memory profiling
try:
    from memory_profiler import profile
except ImportError:
    def profile(func):
        return func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_chunk(chunk, full_buildings):
    """Process a chunk of buildings"""
    analyzer = SolarAnalyzer()
    return analyzer.process_buildings_vectorized(chunk, full_buildings)

@profile
def run_analysis(candidate_buildings, full_buildings, output_dir='results', n_processes=None):
    """Optimized analysis workflow"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Determine the number of processes
        if n_processes is None:
            n_processes = max(1, mp.cpu_count() - 1)
        
        # Initialize dask client
        client = distributed.Client(n_workers=n_processes)
        logger.info(f"Dask client initialized with {n_processes} workers")

        # Optimize input data
        candidate_buildings = SolarAnalyzer.optimize_memory_usage(candidate_buildings)
        full_buildings = SolarAnalyzer.optimize_memory_usage(full_buildings)

        # Convert to dask dataframe
        ddf = dd.from_pandas(candidate_buildings, npartitions=n_processes)

        # Process with enhanced error handling
        results = ddf.map_partitions(
            process_chunk,
            full_buildings=full_buildings,
            meta=get_output_schema()
        ).compute()

        # Save results
        save_results(results, output_path, timestamp)

        return results

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
    finally:
        client.close()
        logger.info("Dask client closed.")

def get_output_schema():
    """Define output schema for dask operations"""
    return pd.DataFrame({
        'solar_potential': pd.Series(dtype='float64'),
        'shadow_factor': pd.Series(dtype='float64'),
        'effective_area': pd.Series(dtype='float64'),
        'peak_power': pd.Series(dtype='float64'),
        'annual_radiation': pd.Series(dtype='float64'),
        'performance_ratio': pd.Series(dtype='float64')
    })

def save_results(results, output_path, timestamp):
    """Save results with optimization"""
    results = SolarAnalyzer.optimize_memory_usage(results)

    # Save to Parquet with gzip compression
    results.to_parquet(
        output_path / f'solar_analysis_results_{timestamp}.parquet',
        compression='gzip'
    )
    logger.info(f"Results saved to {output_path / f'solar_analysis_results_{timestamp}.parquet'}")

# Entry point for script execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run solar analysis.')
    parser.add_argument('candidate_file', help='Path to candidate buildings file')
    parser.add_argument('full_buildings_file', help='Path to full buildings file')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--n-processes', type=int, help='Number of processes to use')

    args = parser.parse_args()

    # Load data
    candidate_buildings = gpd.read_file(args.candidate_file)
    full_buildings = gpd.read_file(args.full_buildings_file)

    # Run analysis
    run_analysis(candidate_buildings, full_buildings, args.output_dir, args.n_processes)