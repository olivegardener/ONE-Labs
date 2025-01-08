import geopandas as gpd
import rasterio
import os
from pathlib import Path
import warnings
import multiprocessing as mp
import time
from datetime import timedelta
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Input folder path (relative to script location)
INPUT_FOLDER = Path(__file__).parent / 'input'

# Target CRS
TARGET_CRS = 'EPSG:6539'

# Minimum parking lot area (sq ft)
MIN_PARKING_AREA = 5000

# Dictionary of data files
data_files = {
    'nsi': {'path': INPUT_FOLDER / 'NYC_NSI.geojson', 'columns': None},
    'lots': {'path': INPUT_FOLDER / 'MapPLUTO.geojson', 'columns': None},
    'pofw': {'path': INPUT_FOLDER / 'NYC_POFW.geojson', 'columns': None},
    'buildings': {'path': INPUT_FOLDER / 'NYC_Buildings.geojson', 'columns': None},
    'facilities': {'path': INPUT_FOLDER / 'NYC_Facilities.geojson', 'columns': None},
}

# Initialize empty dictionaries and lists
datasets = {}
temp_files = []
# Function to print dataset information
def print_dataset_info(name, data):
    print(f"\n=== {name} Dataset ===")
    if isinstance(data, gpd.GeoDataFrame):
        print("\nColumns:")
        for col in data.columns:
            print(f"- {col}")
        print("\nCRS Information:")
        print(data.crs)
        print(f"Number of features: {len(data)}")
    elif isinstance(data, rasterio.DatasetReader):
        print("\nRaster Summary:")
        print(f"Width: {data.width}")
        print(f"Height: {data.height}")
        print(f"Bands: {data.count}")
        print(f"Bounds: {data.bounds}")
        print("\nCRS Information:")
        print(data.crs)
    else:
        print("No data loaded or invalid data format.")

# Function to load and reproject GeoJSON if needed
def load_geojson(file_path, columns=None):
    try:
        gdf = gpd.read_file(
            file_path,
            columns=columns,
            engine='pyogrio'
        )

        # Check if reprojection is needed
        if gdf.crs is None:
            print(f"Warning: {file_path.name} has no CRS defined. Setting to {TARGET_CRS}")
            gdf.set_crs(TARGET_CRS, inplace=True)
        elif gdf.crs.to_string() != TARGET_CRS:
            print(f"Reprojecting {file_path.name} from {gdf.crs} to {TARGET_CRS}")
            gdf = gdf.to_crs(TARGET_CRS)

        return gdf
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to load and reproject raster if needed
def load_raster(file_path):
    try:
        with rasterio.open(file_path) as src:
            # Check if reprojection is needed
            if src.crs.to_string() != TARGET_CRS:
                print(f"Reprojecting {file_path.name} from {src.crs} to {TARGET_CRS}")

                transform, width, height = calculate_default_transform(
                    src.crs, TARGET_CRS, src.width, src.height, *src.bounds)

                temp_path = file_path.parent / f"temp_{file_path.name}"

                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': TARGET_CRS,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                with rasterio.open(temp_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=TARGET_CRS,
                            resampling=Resampling.bilinear
                        )

                return rasterio.open(temp_path)
            else:
                return rasterio.open(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Geometry handling functions
def clean_and_validate_geometry(gdf):
    """Clean and validate geometries in a GeoDataFrame."""
    # Make valid geometries and buffer by tiny amount to fix topology
    gdf.geometry = gdf.geometry.make_valid().buffer(0.01).buffer(-0.01)

    # Remove empty or invalid geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid]

    # Ensure all geometries are polygons or multipolygons
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    return gdf

def explode_multipart_features(gdf):
    """Explode multipart features into single part features."""
    # Explode multipart geometries
    exploded = gdf.explode(index_parts=True)

    # Reset index and drop the multiindex parts
    exploded = exploded.reset_index(drop=True)

    return exploded

def safe_dissolve(gdf, dissolve_field):
    """Safely dissolve geometries while handling topology."""
    try:
        # Group by dissolve field
        groups = gdf.groupby(dissolve_field)
        dissolved_parts = []

        for name, group in groups:
            # Clean group geometries
            group = clean_and_validate_geometry(group)

            if len(group) > 0:
                # Union geometries within group
                unified = group.geometry.unary_union

                # Create new GeoDataFrame with dissolved geometry
                dissolved_part = gpd.GeoDataFrame(
                    {dissolve_field: [name]},
                    geometry=[unified],
                    crs=group.crs
                )

                # Explode any multipart features
                dissolved_part = explode_multipart_features(dissolved_part)
                dissolved_parts.append(dissolved_part)

        # Combine all dissolved parts
        if dissolved_parts:
            result = pd.concat(dissolved_parts, ignore_index=True)
            return clean_and_validate_geometry(result)
        return gdf

    except Exception as e:
        print(f"Error during dissolve operation: {e}")
        return gdf

def process_facilities(gdf):
    print("\nPrepare NYC Facilities...")
    # Add RH_Priority column
    gdf['RH_Priority'] = None
    # Priority 1 facilities
    priority1_mask = (gdf['FACTYPE'] == 'PUBLIC LIBRARY') | \
                    (gdf['FACTYPE'].str.startswith('NYCHA COMMUNITY CENTER', na=False))
    gdf.loc[priority1_mask, 'RH_Priority'] = 1
    # Priority 2 facilities
    priority2_types = [
        'OTHER SCHOOL - NON-PUBLIC', 'COMMUNITY SERVICES', 'ELEMENTARY SCHOOL - PUBLIC',
        'FOOD PANTRY', 'HIGH SCHOOL - PUBLIC', 'CHARTER SCHOOL', 'COMPASS ELEMENTARY',
        'LICENSED PRIVATE SCHOOLS', 'SENIOR CENTER', 
        'JUNIOR HIGH-INTERMEDIATE-MIDDLE SCHOOL - PUBLIC',
        'AFTERSCHOOL PROGRAMS; COMMUNITY SERVICES; EDUCATIONAL SERVICES; FAMILY SUPPORT',
        'HOSPITAL EXTENSION CLINIC', 'FIREHOUSE', 'NURSING HOME', 'K-8 SCHOOL - PUBLIC',
        'COMMUNITY SERVICES; FAMILY SUPPORT; HOUSING SUPPORT; IMMIGRANT SERVICES',
        'SENIOR SERVICES', 'PRE-K CENTER', 'SOUP KITCHEN', 'ELEMENTARY SCHOOL - CHARTER',
        'K-8 SCHOOL - CHARTER', 'PRE-SCHOOL FOR STUDENTS WITH DISABILITIES', 'HOSPITAL',
        'ADULT HOME', 'SENIORS', 
        'AFTERSCHOOL PROGRAMS; COMMUNITY SERVICES; EDUCATIONAL SERVICES; FAMILY SUPPORT; IMMIGRANT SERVICES',
        'K-12 ALL GRADES SCHOOL - PUBLIC, SPECIAL EDUCATION', 'HIGH SCHOOL - CHARTER',
        'K-12 ALL GRADES SCHOOL - CHARTER', 'JUNIOR HIGH-INTERMEDIATE-MIDDLE SCHOOL',
        'SECONDARY SCHOOL - CHARTER', 'BOROUGH OFFICE'
    ]
    priority2_mask = gdf['FACTYPE'].isin(priority2_types)
    gdf.loc[priority2_mask, 'RH_Priority'] = 2
    # Drop rows without priority
    gdf = gdf[gdf['RH_Priority'].notna()].copy()
    # Create and populate fclass field
    gdf['name'] = gdf['FACNAME'].str.title()
    # Assign fclass values
    school_types = [
        'OTHER SCHOOL - NON-PUBLIC', 'ELEMENTARY SCHOOL - PUBLIC',
        'SECONDARY SCHOOL - CHARTER', 'HIGH SCHOOL - PUBLIC', 'CHARTER SCHOOL',
        'COMPASS ELEMENTARY', 'LICENSED PRIVATE SCHOOLS',
        'JUNIOR HIGH-INTERMEDIATE-MIDDLE SCHOOL - PUBLIC', 'K-8 SCHOOL - PUBLIC',
        'PRE-K CENTER', 'ELEMENTARY SCHOOL - CHARTER', 'K-8 SCHOOL - CHARTER',
        'PRE-SCHOOL FOR STUDENTS WITH DISABILITIES',
        'K-12 ALL GRADES SCHOOL - PUBLIC, SPECIAL EDUCATION',
        'HIGH SCHOOL - CHARTER', 'K-12 ALL GRADES SCHOOL - CHARTER',
        'JUNIOR HIGH-INTERMEDIATE-MIDDLE SCHOOL'
    ]
    conditions = [
        (gdf['FACTYPE'].isin(school_types)),
        (gdf['FACTYPE'].str.contains('COMMUNITY SERVICES', na=False)),
        (gdf['FACTYPE'] == 'PUBLIC LIBRARY'),
        (gdf['FACTYPE'] == 'NYCHA COMMUNITY CENTER')
    ]
    choices = ['School', 'Community Services', 'Public Library', 'NYCHA Community Center']
    gdf['fclass'] = np.select(conditions, choices, default=gdf['FACTYPE'].str.title())
    # Keep only required columns
    keep_cols = ['geometry', 'fclass', 'name', 'FACTYPE', 'FACSUBGRP', 'FACGROUP',
                'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'RH_Priority']
    gdf = gdf[keep_cols]
    
    # Remove rows where fclass is 'Community Facilities' and name contains 'State Park'
    # gdf = gdf[~((gdf['fclass'] == 'Community Facilities') & 
    #             (gdf['name'].str.contains('State Park', na=False)))]
    
    # Print summaries
    print("\nFacility counts by fclass:")
    print(gdf['fclass'].value_counts())
    print(f"\nTotal number of facilities: {len(gdf)}")
    return gdf

def process_pofw(gdf):
    print("\nPrepare places of worship...")

    # Keep only required columns and add prefix to fclass
    if 'fclass' not in gdf.columns:
        gdf['fclass'] = 'pofw'
    gdf['fclass'] = 'pofw_' + gdf['fclass'].astype(str)

    # Ensure name column exists
    if 'name' not in gdf.columns:
        gdf['name'] = 'Unknown'

    # Add new columns with default values
    text_columns = ['FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN', 
                   'OPNAME', 'OPABBREV', 'OPTYPE']
    for col in text_columns:
        gdf[col] = 'Unknown'

    # Add numeric columns
    gdf['CAPACITY'] = 0
    gdf['RH_Priority'] = 1

    # Fill NA values
    gdf['name'] = gdf['name'].fillna('Unknown')

    # Keep only required columns
    keep_cols = ['geometry', 'fclass', 'name', 'FACTYPE', 'FACSUBGRP', 'FACGROUP',
                'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'RH_Priority']
    gdf = gdf[keep_cols]

    print(f"Number of points in POFW dataset: {len(gdf)}")

    return gdf

def merge_point_datasets(datasets):
    print("\nMerge points datasets...")

    required_columns = ['fclass', 'name', 'FACTYPE', 'FACSUBGRP', 'FACGROUP',
                       'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 
                       'RH_Priority', 'geometry']

    # Verify columns in each dataset
    points_dfs = []
    for key in ['pofw', 'facilities']:
        if key in datasets and datasets[key] is not None:
            df = datasets[key]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"Warning: {key} is missing columns: {missing_cols}")
                continue
            points_dfs.append(df[required_columns])

    if not points_dfs:
        print("No valid datasets to merge!")
        return gpd.GeoDataFrame(columns=required_columns, crs=TARGET_CRS)

    # Merge datasets
    bldg_pts = pd.concat(points_dfs, ignore_index=True)

    # Add ObjectID
    bldg_pts['ObjectID'] = bldg_pts.index + 1

    # Print summary information
    print("\nColumns in merged dataset:", bldg_pts.columns.tolist())
    print("\nCount by fclass:")
    print(bldg_pts['fclass'].value_counts())
    print(f"\nTotal building points: {len(bldg_pts)}")

    return gpd.GeoDataFrame(bldg_pts, geometry='geometry', crs=TARGET_CRS)

# Main processing block
if not INPUT_FOLDER.exists():
    raise FileNotFoundError(f"Input folder not found at: {INPUT_FOLDER}")

n_cores = max(1, mp.cpu_count() - 1)

try:
    total_start_time = time.time()

    # Load all datasets
    for key, file_info in data_files.items():
        file_path = file_info['path']
        columns = file_info['columns']

        start_time = time.time()
        print(f"\nLoading {file_path.name}...")
        if file_path.suffix == '.geojson':
            datasets[key] = load_geojson(file_path, columns)
        elif file_path.suffix == '.tif':
            datasets[key] = load_raster(file_path)
            if datasets[key] is not None and datasets[key].name != str(file_path):
                temp_files.append(Path(datasets[key].name))

        end_time = time.time()
        duration = end_time - start_time
        print(f"Loading time for {key}: {timedelta(seconds=duration)}")

        # Print immediate summary after loading each dataset
        if datasets[key] is not None:
            print(f"CRS for {key}: {datasets[key].crs}")
            if isinstance(datasets[key], gpd.GeoDataFrame):
                print(f"{key} loaded with {len(datasets[key])} features.")
            else:
                print(f"{key} loaded but no recognizable format.")
        else:
            print(f"{key} dataset could not be loaded or is empty.")

    # Process facilities
    if 'facilities' in datasets and datasets['facilities'] is not None:
        datasets['facilities'] = process_facilities(datasets['facilities'])

    # Process POFW
    if 'pofw' in datasets and datasets['pofw'] is not None:
        datasets['pofw'] = process_pofw(datasets['pofw'])

    # Merge points datasets
    bldg_pts = merge_point_datasets(datasets)

    # Extract data from lots
    print("\nExtract data from lots...")
    lots = datasets['lots']
    if lots is not None and len(bldg_pts) > 0:
        lot_fields = ['Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 'ComArea',
                     'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea',
                     'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR']
        bldg_pts = gpd.sjoin(bldg_pts, lots[lot_fields + ['geometry']], how='left', predicate='within')
        if 'index_right' in bldg_pts.columns:
            bldg_pts.drop(columns=['index_right'], inplace=True)

        matched = bldg_pts[~bldg_pts['Address'].isna()].shape[0]
        unmatched = bldg_pts[bldg_pts['Address'].isna()].shape[0]
        print(f"Points successfully extracted data from lots: {matched}")
        print(f"Points did not extract data from lots: {unmatched}")

    # Combine points data with building footprints
    print("\nCombine points data with building footprints...")
    buildings = datasets['buildings']
    if buildings is not None and len(bldg_pts) > 0:
        buildings = buildings[['geometry', 'groundelev', 'heightroof', 'lststatype', 'cnstrct_yr']]
        joined = gpd.sjoin(buildings, bldg_pts, how='inner', predicate='contains')
        print("Columns in joined after sjoin:", joined.columns)

        def combine_values(series):
            vals = series.dropna().unique()
            return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan

        agg_fields = [
            'fclass', 'name', 'Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 
            'ComArea', 'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea', 
            'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR', 'FACTYPE', 'FACSUBGRP', 
            'FACGROUP', 'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'RH_Priority'
        ]

        agg_fields = [f for f in agg_fields if f in joined.columns]
        if agg_fields:
            grouped = joined.groupby(joined.index)
            agg_dict = {f: combine_values for f in agg_fields}
            # Special handling for numeric fields
            if 'CAPACITY' in joined.columns:
                agg_dict['CAPACITY'] = 'sum'
            if 'RH_Priority' in joined.columns:
                agg_dict['RH_Priority'] = 'min'  # Take the highest priority (lowest number)

            agg_result = grouped.agg(agg_dict)
            buildings = buildings.loc[agg_result.index]
            for f in agg_fields:
                buildings[f] = agg_result[f]

        datasets['buildings'] = buildings
    else:
        print("No buildings or points to combine or no overlapping features found.")

    # Extract data from NSI
    print("\nExtract data from national structures inventory...")
    nsi = datasets['nsi']
    if nsi is not None and buildings is not None:
        nsi_fields = ['bldgtype', 'num_story', 'found_type', 'found_ht']
        joined_nsi = gpd.sjoin(buildings, nsi[nsi_fields + ['geometry']], how='left', predicate='contains')

        def combine_values(series):
            vals = series.dropna().unique()
            return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan

        agg_dict_nsi = {f: combine_values for f in nsi_fields}
        grouped_nsi = joined_nsi.groupby(joined_nsi.index).agg(agg_dict_nsi)
        for f in nsi_fields:
            buildings[f] = grouped_nsi[f]

        matched_nsi = buildings[~buildings['bldgtype'].isna()].shape[0]
        unmatched_nsi = buildings[buildings['bldgtype'].isna()].shape[0]
        print(f"Buildings successfully extracted data from nsi: {matched_nsi}")
        print(f"Buildings did not extract data from nsi: {unmatched_nsi}")

    # # Isolate parking lots
    # print("\nIsolate parking lots from PLUTO data...")
    # if lots is not None:
    #     try:
    #         lots['LandUse'] = lots['LandUse'].astype(str)
    #         parking = lots[lots['LandUse'] == '10'].copy()

    #         parking = clean_and_validate_geometry(parking)
    #         parking = parking[parking['LotArea'] >= MIN_PARKING_AREA].copy()

    #         parking['fclass'] = "Parking"
    #         parking['name'] = parking['Address']

    #         # Add the new fields to match the building dataset
    #         parking['FACTYPE'] = 'Parking Lot'
    #         parking['FACSUBGRP'] = 'Transportation'
    #         parking['FACGROUP'] = 'Transportation Infrastructure'
    #         parking['FACDOMAIN'] = 'Transportation'
    #         parking['OPNAME'] = parking['OwnerName']
    #         parking['OPABBREV'] = 'Unknown'
    #         parking['OPTYPE'] = 'Private'
    #         parking['CAPACITY'] = 0
    #         parking['RH_Priority'] = 3

    #         print("Dissolving parking lots...")
    #         parking = safe_dissolve(parking, 'OwnerName')
    #         parking = explode_multipart_features(parking)
    #         parking = clean_and_validate_geometry(parking)

    #         print(f"Number of parking lots and structures (>= {MIN_PARKING_AREA} sq ft): {len(parking)}")

    #         invalid_count = sum(~parking.geometry.is_valid)
    #         if invalid_count > 0:
    #             print(f"Warning: {invalid_count} invalid geometries found after processing")
    #             parking = parking[parking.geometry.is_valid]

    #     except Exception as e:
    #         print(f"Error processing parking lots: {e}")
    #         parking = gpd.GeoDataFrame(columns=['fclass', 'name', 'geometry'], crs=TARGET_CRS)
    # else:
    #     parking = gpd.GeoDataFrame(columns=['fclass', 'name', 'geometry'], crs=TARGET_CRS)

    # # Merge parking lots and buildings
    # print("\nMerge parking lots and buildings...")
    # if buildings is not None:
    #     if 'fclass' not in buildings.columns:
    #         buildings['fclass'] = 'Building'
    # else:
    #     buildings = gpd.GeoDataFrame(
    #         columns=['fclass', 'geometry'] + ['FACTYPE', 'FACSUBGRP', 'FACGROUP', 'FACDOMAIN', 
    #                                         'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'RH_Priority'],
    #         crs=TARGET_CRS
    #     )
    #     buildings['fclass'] = 'Building'

    # # Ensure all required columns are present in both datasets
    # required_cols = ['fclass', 'name', 'geometry', 'FACTYPE', 'FACSUBGRP', 'FACGROUP', 
    #                 'FACDOMAIN', 'OPNAME', 'OPABBREV', 'OPTYPE', 'CAPACITY', 'RH_Priority']

    # for df in [parking, buildings]:
    #     for col in required_cols:
    #         if col not in df.columns and col != 'geometry':
    #             df[col] = 'Unknown' if col not in ['CAPACITY', 'RH_Priority'] else 0

    # # Merge them together
    # sites = pd.concat([buildings, parking], ignore_index=True)
    # print(f"Number of sites of interest: {len(sites)}")

    # Create output directory if it doesn't exist
    output_dir = Path("./output/sitesofinterest")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export final sites
    buildings.to_file(output_dir / "preprocessed_sites_RH.shp", driver="ESRI Shapefile")
    buildings.to_file("./output/preprocessed_sites_RH.geojson", driver="GeoJSON")

    total_duration = time.time() - total_start_time
    print(f"\nTotal processing time: {timedelta(seconds=total_duration)}")

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Close raster datasets
    for name, dataset in datasets.items():
        if isinstance(dataset, rasterio.DatasetReader):
            dataset.close()

    # Clean up temporary files
    for temp_file in temp_files:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {e}")
