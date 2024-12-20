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

# Function to load and reproject GeoJSON if needed
def load_geojson(file_path, columns=None):
	try:
		# Use memory mapping for large files
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

				# Calculate the transform for the new projection
				transform, width, height = calculate_default_transform(
					src.crs, TARGET_CRS, src.width, src.height, *src.bounds)

				# Create a temporary file for the reprojected raster
				temp_path = file_path.parent / f"temp_{file_path.name}"

				# Create the reprojected raster
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

				# Return the reprojected raster
				return rasterio.open(temp_path)
			else:
				# If no reprojection needed, return the original
				return rasterio.open(file_path)
	except Exception as e:
		print(f"Error loading {file_path}: {e}")
		return None

# Dictionary to store file paths and their essential columns (if known)
data_files = {
	'nsi': {'path': INPUT_FOLDER / 'NYC_NSI.geojson', 'columns': None},
	'firehouses': {'path': INPUT_FOLDER / 'FDNY_Firehouse.geojson', 'columns': None},
	'lots': {'path': INPUT_FOLDER / 'MapPLUTO.geojson', 'columns': None},
	'pofw': {'path': INPUT_FOLDER / 'NYC_POFW.geojson', 'columns': None},
	'buildings': {'path': INPUT_FOLDER / 'NYC_Buildings.geojson', 'columns': None},
	'pois': {'path': INPUT_FOLDER / 'NYC_POIS.geojson', 'columns': None},
	'libraries': {'path': INPUT_FOLDER / 'NYC_Libraries.geojson', 'columns': None},
	'schools': {'path': INPUT_FOLDER / 'NYC_Schools.geojson', 'columns': None},
}

# Dictionary to store loaded datasets
datasets = {}
temp_files = []  # To track temporary reprojected files

# Check if input folder exists
if not INPUT_FOLDER.exists():
	raise FileNotFoundError(f"Input folder not found at: {INPUT_FOLDER}")

# Set number of CPU cores to use (leave one core free)
n_cores = max(1, mp.cpu_count() - 1)

try:
	# Start total timer
	total_start_time = time.time()

	# Load all datasets
	for key, file_info in data_files.items():
		file_path = file_info['path']
		columns = file_info['columns']

		# Start timer for this dataset
		start_time = time.time()
		print(f"\nLoading {file_path.name}...")

		if file_path.suffix == '.geojson':
			datasets[key] = load_geojson(file_path, columns)
		elif file_path.suffix == '.tif':
			datasets[key] = load_raster(file_path)
			if datasets[key] is not None and datasets[key].name != str(file_path):
				temp_files.append(Path(datasets[key].name))

		# End timer for this dataset
		end_time = time.time()
		duration = end_time - start_time
		print(f"Loading time: {timedelta(seconds=duration)}")

	# Print information for all datasets
	print("\n=== DATASET SUMMARIES ===")
	for name, data in datasets.items():
		print_dataset_info(name, data)

	# ---------------------------------------------------------
	# Prepare places of interest (POIs)
	# ---------------------------------------------------------
	print("\nPrepare places of interest...")
	pois = datasets['pois']
	if pois is not None:
		print(f"Number of points in original POIs dataset: {len(pois)}")
		# Filter for typologies of interest
		typologies = ['arts_centre', 'college', 'community_centre', 'shelter']
		pois = pois[pois['fclass'].isin(typologies)].copy()
		# Keep only geometry, fclass, name
		keep_cols = ['fclass', 'name', 'geometry']
		pois = pois[keep_cols]
		print(f"Number of points in filtered POIs dataset: {len(pois)}")
		datasets['pois'] = pois

	# ---------------------------------------------------------
	# Prepare places of worship (POFW)
	# ---------------------------------------------------------
	print("\nPrepare places of worship...")
	pofw = datasets['pofw']
	if pofw is not None:
		# Prefix fclass with 'pofw_'
		pofw['fclass'] = 'pofw_' + pofw['fclass'].astype(str)
		# Add a new column 'name' from 'Name' or 'Unknown'
		pofw['name'] = pofw['name'].fillna('Unknown')
		# Keep only geometry, fclass, name
		pofw = pofw[['fclass', 'name', 'geometry']]
		print(f"Number of points in filtered POFW dataset: {len(pofw)}")
		datasets['pofw'] = pofw

	# ---------------------------------------------------------
	# Prepare firehouses
	# ---------------------------------------------------------
	print("\nPrepare firehouses...")
	firehouses = datasets['firehouses']
	if firehouses is not None:
		# Add fclass = "Firehouse"
		firehouses['fclass'] = "Firehouse"
		# Add name = FacilityName
		firehouses['name'] = firehouses['FacilityName']
		# Keep only fclass, name, geometry
		firehouses = firehouses[['fclass', 'name', 'geometry']]
		print(f"Number of firehouses in prepared dataset: {len(firehouses)}")
		datasets['firehouses'] = firehouses

	# ---------------------------------------------------------
	# Prepare libraries
	# ---------------------------------------------------------
	print("\nPrepare libraries...")
	libraries = datasets['libraries']
	if libraries is not None:
		# fclass = "Library"
		libraries['fclass'] = "Library"
		# name = NAME
		libraries['name'] = libraries['NAME']
		# Keep only fclass, name, geometry
		libraries = libraries[['fclass', 'name', 'geometry']]
		print(f"Number of libraries in prepared dataset: {len(libraries)}")
		datasets['libraries'] = libraries

	# ---------------------------------------------------------
	# Prepare schools
	# ---------------------------------------------------------
	print("\nPrepare schools...")
	schools = datasets['schools']
	if schools is not None:
		# fclass = "School"
		schools['fclass'] = "School"
		# name = Name
		schools['name'] = schools['Name']
		# Keep only fclass, name, geometry
		schools = schools[['fclass', 'name', 'geometry']]
		print(f"Number of schools in prepared dataset: {len(schools)}")
		datasets['schools'] = schools

	# ---------------------------------------------------------
	# Merge Points Datasets
	# ---------------------------------------------------------
	print("\nMerge points datasets...")
	# Datasets to merge: pois, pofw, firehouses, libraries, schools
	points_dfs = [datasets.get(k) for k in ['pois', 'pofw', 'firehouses', 'libraries', 'schools'] if datasets.get(k) is not None]

	# Verify required columns
	required_columns = ['fclass', 'name', 'geometry']
	for df in points_dfs:
		if not all(col in df.columns for col in required_columns):
			raise ValueError("One of the point datasets does not contain the required columns.")

	# Concatenate
	if len(points_dfs) > 0:
		bldg_pts = pd.concat(points_dfs, ignore_index=True)[required_columns]

		# Add ObjectID
		bldg_pts['ObjectID'] = bldg_pts.index + 1

		# Print columns
		print("Columns in merged dataset:", bldg_pts.columns.tolist())

		# Print count by fclass
		print("\nCount by fclass:")
		print(bldg_pts['fclass'].value_counts())

		# Print total count
		print(f"Total building points: {len(bldg_pts)}")
	else:
		bldg_pts = gpd.GeoDataFrame(columns=required_columns + ['ObjectID'], crs=TARGET_CRS)

	# Ensure CRS is set
	bldg_pts = gpd.GeoDataFrame(bldg_pts, geometry='geometry', crs=TARGET_CRS)

	# ---------------------------------------------------------
	# Extract data from lots
	# ---------------------------------------------------------
	print("\nExtract data from lots...")
	lots = datasets['lots']
	if lots is not None and len(bldg_pts) > 0:
		# Spatial Join
		# Fields to extract
		lot_fields = ['Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 'ComArea',
		              'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea',
		              'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR']
		bldg_pts = gpd.sjoin(bldg_pts, lots[lot_fields + ['geometry']], how='left', predicate='within')
		# After join, drop the index_right column
		if 'index_right' in bldg_pts.columns:
			bldg_pts.drop(columns=['index_right'], inplace=True)

		matched = bldg_pts[~bldg_pts['Address'].isna()].shape[0]
		unmatched = bldg_pts[bldg_pts['Address'].isna()].shape[0]

		print(f"Points successfully extracted data from lots: {matched}")
		print(f"Points did not extract data from lots: {unmatched}")

	# ---------------------------------------------------------
	# Combine points data with building footprints
	# ---------------------------------------------------------
	print("\nCombine points data with building footprints...")
	buildings = datasets['buildings']
	if buildings is not None and len(bldg_pts) > 0:
		# From ‘buildings’, keep only geometry and the specified fields
		buildings = buildings[['geometry', 'groundelev', 'heightroof', 'lststatype', 'cnstrct_yr']]

		# Perform an inner spatial join between buildings and bldg_pts
		joined = gpd.sjoin(buildings, bldg_pts, how='inner', predicate='contains')
		print("Columns in joined after sjoin:", joined.columns)

		# Fields to aggregate
		agg_fields = [
			'fclass', 'name', 'Address', 'BldgClass', 'OwnerName', 'LotArea', 'BldgArea', 
			'ComArea', 'ResArea', 'OfficeArea', 'RetailArea', 'GarageArea', 'StrgeArea', 
			'BuiltFAR', 'ResidFAR', 'CommFAR', 'FacilFAR'
		]

		def combine_values(series):
			vals = series.dropna().unique()
			return ' + '.join(map(str, vals)) if len(vals) > 0 else np.nan

		# Filter agg_fields to those present in joined
		agg_fields = [f for f in agg_fields if f in joined.columns]

		if agg_fields:
			# Group by building index (the index of 'buildings' in the join)
			grouped = joined.groupby(joined.index)
			agg_dict = {f: combine_values for f in agg_fields}
			agg_result = grouped.agg(agg_dict)

			# Filter buildings to only those that matched points
			buildings = buildings.loc[agg_result.index]

			# Merge aggregated results back into buildings
			for f in agg_fields:
				buildings[f] = agg_result[f]

		datasets['buildings'] = buildings
	else:
		print("No buildings or points to combine or no overlapping features found.")

	# ---------------------------------------------------------
	# Extract data from national structures inventory (nsi)
	# ---------------------------------------------------------
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

	# ---------------------------------------------------------
	# Isolate parking lots from PLUTO data
	# ---------------------------------------------------------
	print("\nIsolate parking lots from PLUTO data...")
	if lots is not None:
		# LandUse = 10 means parking
		lots['LandUse'] = lots['LandUse'].astype(str)
		parking = lots[lots['LandUse'] == '10'].copy()
		parking['fclass'] = "Parking"
		parking['name'] = parking['Address']

		# Dissolve by OwnerName (if desired to group contiguous lots)
		parking = parking.dissolve(by='OwnerName', as_index=False)

		print(f"Number of parking lots and structures: {len(parking)}")
	else:
		parking = gpd.GeoDataFrame(columns=['fclass', 'name', 'geometry'], crs=TARGET_CRS)

	# ---------------------------------------------------------
	# Merge parking lots and buildings
	# ---------------------------------------------------------
	print("\nMerge parking lots and buildings...")

	# Align fields
	bldg_fields = set(buildings.columns)
	parking_fields = set(parking.columns)
	extra_in_parking = parking_fields - bldg_fields
	parking = parking.drop(columns=list(extra_in_parking), errors='ignore')

	missing_in_parking = bldg_fields - set(parking.columns)
	for f in missing_in_parking:
		if f == 'geometry':
			continue
		parking[f] = 'Unknown'

	parking = parking[buildings.columns]

	sites = pd.concat([buildings, parking], ignore_index=True)
	print(f"Number of sites of interest: {len(sites)}")

	# ---------------------------------------------------------
	# export to shapefile and geojson
	# ---------------------------------------------------------
	sites.to_file("./output/sitesofinterest/preprocessed_sites.shp", driver="ESRI Shapefile")
	sites.to_file("./output/preprocessed_sites.geojson", driver="GeoJSON")

	# ---------------------------------------------------------
	# Print total time
	# ---------------------------------------------------------
	total_duration = time.time() - total_start_time
	print(f"\nTotal processing time: {timedelta(seconds=total_duration)}")

except Exception as e:
	print(f"Error occurred: {e}")

finally:
	# Close raster datasets and clean up temporary files
	for name, dataset in datasets.items():
		if isinstance(dataset, rasterio.DatasetReader):
			dataset.close()

	# Clean up temporary reprojected files
	for temp_file in temp_files:
		if temp_file.exists():
			try:
				temp_file.unlink()
			except Exception as e:
				print(f"Error removing temporary file {temp_file}: {e}")