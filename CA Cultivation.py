# %% Imports
import datetime
import itertools
     
# Built-in modules
import os
    
# Basics of Python data handling and visualization
import numpy as np
from aenum import MultiValueEnum

np.random.seed(42)
import geopandas as gpd
import joblib

# Machine learning
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import Polygon
from sklearn import metrics, preprocessing
from tqdm import tqdm

from sentinelhub import DataCollection, UtmZoneSplitter, BBoxSplitter, CRS, BBox

# Imports from eo-learn and sentinelhub-py
from eolearn.core import (
    EOExecutor,
    EOPatch,
    EOTask,
    EOWorkflow,
    FeatureType,
    LoadTask,
    MergeFeatureTask,
    OverwritePermission,
    SaveTask,
    linearly_connect_tasks,
)
from eolearn.features import LinearInterpolationTask, NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.geometry import ErosionTask, VectorToRasterTask
from eolearn.io import ExportToTiffTask, SentinelHubInputTask, VectorImportTask
from eolearn.ml_tools import FractionSamplingTask

from shapely.ops import polygonize

# %% Creating and sorting reference LULC map


#bbox = (-121.7107,36.5494,-121.5102,36.7411)

bbox = (-119.998169,36.424598,-119.505157,36.842812)

ref = gpd.read_file("C:/Users/bdove/Downloads/i15_crop_mapping_2019/i15_Crop_Mapping_2019/i15_Crop_Mapping_2019.shp", bbox)

#Adding cultivated column to easily sort later                    
#ref['Cultivated'] = 0



ref.to_file('C:/Users/bdove/Desktop/LULC Project/CA LULC/selreflulc.gpkg', driver="GPKG")

#Converting to 4326 so Sentinelhub can read it
#df = df.to_crs("EPSG:4326")

#%% 

ref = gpd.read_file('C:/Users/bdove/Desktop/LULC Project/CA LULC/selreflulc.gpkg')



# %% Reading and processing CA border data

# Folder where data for running the notebook is stored
DATA_FOLDER = 'C:/Users/bdove/Desktop/LULC Project/CA LULC/eo-learn-master/example_data'
# Locations for collected data and intermediate results
EOPATCH_FOLDER = 'C:/Users/bdove/Desktop/LULC Project/CA LULC/eopatches'
EOPATCH_SAMPLES_FOLDER = 'C:/Users/bdove/Desktop/LULC Project/CA LULC/eopatches_sampled'
RESULTS_FOLDER = 'C:/Users/bdove/Desktop/LULC Project/CA LULC/results'

for folder in (EOPATCH_FOLDER, EOPATCH_SAMPLES_FOLDER, RESULTS_FOLDER):
    os.makedirs(folder, exist_ok=True)


from shapely.geometry import box
#coords for selinas
#selshape = box(-121.7107,36.5494,-121.5102,36.7411)

#shape of fresno
selshape = box(-119.998169,36.424598,-119.505157,36.842812)
df = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[selshape])
print(df)


"""
df.plot()
plt.axis("off")
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/shape.png")


#Get the country's shape in polygon format
selshape = df.geometry.values[0]
sel_bord = co_bord.buffer(500)



#Splitting into 9x12.5km boxes
#bbox_splitter = BBoxSplitter([coshape], CRS.WGS84, (50,50))
"""





# %%
bbox_splitter = BBoxSplitter([selshape], CRS.WGS84, (20,20))

"""
# Reversing reordering lng-lat into lat-long to sentinelhub can read the request
bbox_list = bbox_splitter.get_bbox_list()



for i in range(len(bbox_list)):
    #bbox_list.reverse() -> reverse function doesn't work for some reason, doing it manually:
    oldminx = bbox_list[i].min_x
    oldminy = bbox_list[i].min_y
    oldmaxx = bbox_list[i].max_x
    oldmaxy = bbox_list[i].max_y 
    bbox_list[i].min_x = oldminy
    bbox_list[i].min_y = oldminx
    bbox_list[i].max_x = oldmaxy
    bbox_list[i].max_y = oldmaxx
"""    
bbox_list = np.array(bbox_splitter.get_bbox_list())


info_list = bbox_splitter.get_info_list()

"""
# Reversing reordering lng-lat into lat-long to sentinelhub can read the request
for i in range(len(info_list)):
    il = info_list[i]
    ilr = il['parent_bbox']
    ilr = ilr.reverse()
    info_list[i].update({'parent_bbox': ilr})
"""



#Adding index key to info_list so bboxes can be labeled properly
idf = []
for i in range(len(info_list)):
    newil = info_list[i]
    newil['index'] = i
    idf.append(newil)
    
info_list = np.array(idf)


geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]


idxs = [info["index"] for info in info_list]
idxs_x = [info["index_x"] for info in info_list]
idxs_y = [info["index_y"] for info in info_list]



bbox_gdf = gpd.GeoDataFrame({"index": idxs, "index_x": idxs_x, "index_y": idxs_y}, crs=df.crs, geometry=geometry)



# select a 5x5 area (id of center patch)
ID = 187

# Obtain surrounding 5x5 patches
patchIDs = []
for idx, (bbox, info) in enumerate(zip(bbox_list, info_list)):
    if abs(info["index_x"] - info_list[ID]["index_x"]) <= 2 and abs(info["index_y"] - info_list[ID]["index_y"]) <= 2:
        patchIDs.append(idx)

# Check if final size is 5x5
if len(patchIDs) != 5 * 5:
    print("Warning! Use a different central patch ID, this one is on the border.")



# Change the order of the patches (useful for plotting)
patchIDs = np.transpose(np.fliplr(np.array(patchIDs).reshape(5,5))).ravel()

# Save to shapefile
shapefile_name = "grid_ca_selinas.gpkg"
bbox_gdf.to_file(os.path.join(RESULTS_FOLDER, shapefile_name), driver="GPKG")


# Display bboxes over country
fig, ax = plt.subplots(figsize=(30, 30))
ax.set_title("Selected 5x5 tiles from California", fontsize=25)


df.plot(ax=ax, facecolor="w", edgecolor="black", alpha=0.5)


bbox_gdf.plot(ax=ax, facecolor="w", edgecolor="r", alpha=0.5)



#selshape = box(-121.7107,36.5494,-121.5102,36.7411)
#xlim =(-121.7107, -121.5102)
#ylim =(36.5494,36.7411)
selshape = box(-119.998169,36.424598,-119.505157,36.842812)
xlim = (-119.998169,-119.505157)
ylim = (36.424598, 36.842812)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ref.plot(ax=ax, facecolor="w", edgecolor="b", alpha=0.5)


for bbox, info in zip(bbox_list, info_list):
    geo = bbox.geometry
    ax.text(geo.centroid.x, geo.centroid.y, info["index"], ha="center", va="center")

# Mark bboxes of selected area
bbox_gdf[bbox_gdf.index.isin(patchIDs)].plot(ax=ax, facecolor="g", edgecolor="r", alpha=0.5)

plt.axis("off");
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/selinasbbox.png")

# %% Defining SentinelHub classes & workflow

class SentinelHubValidDataTask(EOTask):
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __init__(self, output_feature):
        self.output_feature = output_feature

    def execute(self, eopatch):
        eopatch[self.output_feature] = eopatch.mask["IS_DATA"].astype(bool) & (~eopatch.mask["CLM"].astype(bool))
        return eopatch


class AddValidCountTask(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """

    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch[FeatureType.MASK_TIMELESS, self.name] = np.count_nonzero(eopatch.mask[self.what], axis=0)
        return eopatch

# BAND DATA
# Add a request for S2 bands.
# Here we also do a simple filter of cloudy scenes (on tile level).
# The s2cloudless masks and probabilities are requested via additional data.
band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]
add_data = SentinelHubInputTask(
    bands_feature=(FeatureType.DATA, "BANDS"),
    bands=band_names,
    resolution=10,
    maxcc=0.8,
    time_difference=datetime.timedelta(minutes=120),
    data_collection=DataCollection.SENTINEL2_L1C,
    additional_data=[(FeatureType.MASK, "dataMask", "IS_DATA"), (FeatureType.MASK, "CLM"), (FeatureType.DATA, "CLP")],
    max_threads=5,
)


# CALCULATING NEW FEATURES
# NDVI: (B08 - B04)/(B08 + B04)
# NDWI: (B03 - B08)/(B03 + B08)
# NDBI: (B11 - B08)/(B11 + B08)
ndvi = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "NDVI"), [band_names.index("B08"), band_names.index("B04")]
)
ndwi = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "NDWI"), [band_names.index("B03"), band_names.index("B08")]
)
ndbi = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "NDBI"), [band_names.index("B11"), band_names.index("B08")]
)


# VALIDITY MASK
# Validate pixels using SentinelHub's cloud detection mask and region of acquisition
add_sh_validmask = SentinelHubValidDataTask((FeatureType.MASK, "IS_VALID"))

# COUNTING VALID PIXELS
# Count the number of valid observations per pixel using valid data mask
add_valid_count = AddValidCountTask("IS_VALID", "VALID_COUNT")

# SAVING TO OUTPUT (if needed)
save = SaveTask(EOPATCH_FOLDER, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

#%% Sorting reference LULC into cultivated/not cultivated

import numpy as np

cultlist = ['G', 'R', 'F', 'P','T', 'D', 'C', 'V']
uclist = ['I', 'S', 'U', 'UR', 'UC', 'UI', 'UV', 'NC', 'NV', 'NR', 'NW', 'NB', 'X', 'YP']
ndlist = ['NS', 'E', 'Z']


refclass = ref['SYMB_CLASS']


iterlist = []

for i in range(len(ref)):
    if refclass[i] in ndlist:
        #ref['Cultivated'] = 0
        iterlist.append(0)
    elif refclass[i] in cultlist:
        #ref['Cultivated'] = 1
        iterlist.append(1)
    elif refclass[i] in uclist:
        #ref['Cultivated'] = 2
        iterlist.append(2)
    else:
        print("error at index ", i, " with classifier ", ref[i])
        
#iterlist = np.array(iterlist)
        
ref['Cultivated'] = iterlist
        
ref.to_file('C:/Users/bdove/Desktop/LULC Project/CA LULC/sortedselreflulc.gpkg', driver = "GPKG")



#%% Colormap

class LULC(MultiValueEnum):
    """Enum class containing basic LULC types"""

    NO_DATA = "No Data", 0, "#ffffff"
    Cult = "Cultivated", 1, "#054907"
    Not_Cultivated = "Not Cultivated", 2, "#964B00"


    @property
    def id(self):
        return self.values[1]

    @property
    def color(self):
        return self.values[2]


# Reference colormap things
lulc_cmap = ListedColormap([x.color for x in LULC], name="lulc_cmap")
lulc_norm = BoundaryNorm([x - 0.5 for x in range(len(LULC) + 1)], lulc_cmap.N)

land_use_ref_path = 'C:/Users/bdove/Desktop/LULC Project/CA LULC/sortedselreflulc.gpkg'

vector_feature = FeatureType.VECTOR_TIMELESS, "LULC_REFERENCE"

vector_import_task = VectorImportTask(vector_feature, land_use_ref_path)

rasterization_task = VectorToRasterTask(
    vector_feature,
    (FeatureType.MASK_TIMELESS, "LULC"),
    values_column="Cultivated",
    raster_shape=(FeatureType.MASK, "IS_DATA"),
    raster_dtype=np.uint8,
)

# Define the workflow
workflow_nodes = linearly_connect_tasks(
    add_data, ndvi, ndwi, ndbi, add_sh_validmask, add_valid_count, vector_import_task, rasterization_task, save
)
workflow = EOWorkflow(workflow_nodes)
"""
# Let's visualize it
#workflow.dependency_graph('C:/Users/bdove/Desktop/LULC Project/workflow.jpg')
"""




#%% Sorting reference LULC into cultivated/not cultivated

import numpy as np

cultlist = ['G', 'R', 'F', 'P','T', 'D', 'C', 'V']
uclist = ['I', 'S', 'U', 'UR', 'UC', 'UI', 'UV', 'NC', 'NV', 'NR', 'NW', 'NB', 'X', 'YP']
ndlist = ['NS', 'E', 'Z']


refclass = ref['SYMB_CLASS']


iterlist = []

for i in range(len(ref)):
    if refclass[i] == 'G':
        #ref['Cultivated'] = 0
        iterlist.append(1)
    elif refclass[i] == 'R':
        #ref['Cultivated'] = 1
        iterlist.append(2)
    elif refclass[i] == 'F':
        #ref['Cultivated'] = 1
        iterlist.append(3)
    elif refclass[i] == 'P':
        #ref['Cultivated'] = 1
        iterlist.append(4)
    elif refclass[i] == 'T':
        #ref['Cultivated'] = 1
        iterlist.append(5)
    elif refclass[i] == 'D':
        #ref['Cultivated'] = 1
        iterlist.append(6)
    elif refclass[i] == 'C':
        #ref['Cultivated'] = 1
        iterlist.append(7)
    elif refclass[i] == 'V':
        #ref['Cultivated'] = 1
        iterlist.append(8)
    elif refclass[i] == 'I':
        #ref['Cultivated'] = 1
        iterlist.append(9)
    elif refclass[i] == 'S':
        #ref['Cultivated'] = 1
        iterlist.append(10)
    elif refclass[i] == 'U':
        #ref['Cultivated'] = 1
        iterlist.append(11)
    elif refclass[i] == 'UR':
        #ref['Cultivated'] = 1
        iterlist.append(12)
    elif refclass[i] == 'UC':
        #ref['Cultivated'] = 1
        iterlist.append(13)
    elif refclass[i] == 'UI':
        #ref['Cultivated'] = 1
        iterlist.append(14)
    elif refclass[i] == 'UV':
        #ref['Cultivated'] = 1
        iterlist.append(15)
    elif refclass[i] == 'NC':
        #ref['Cultivated'] = 1
        iterlist.append(16)
    elif refclass[i] == 'NV':
        #ref['Cultivated'] = 1
        iterlist.append(17)
    elif refclass[i] == 'NR':
        #ref['Cultivated'] = 1
        iterlist.append(18)
    elif refclass[i] == 'NW':
        #ref['Cultivated'] = 1
        iterlist.append(19)
    elif refclass[i] == 'NB':
        #ref['Cultivated'] = 1
        iterlist.append(20)
    elif refclass[i] == 'NS':
        #ref['Cultivated'] = 1
        iterlist.append(21)
    elif refclass[i] == 'E':
        #ref['Cultivated'] = 1
        iterlist.append(22)
    elif refclass[i] == 'Z':
        #ref['Cultivated'] = 1
        iterlist.append(23)
    elif refclass[i] == 'X':
        #ref['Cultivated'] = 1
        iterlist.append(24)
    elif refclass[i] == 'YP':
        #ref['Cultivated'] = 1
        iterlist.append(25)
    else:
        print("error at index ", i, " with classifier ", ref[i])
        
#iterlist = np.array(iterlist)
        
ref['Cultivated'] = iterlist
        
ref.to_file('C:/Users/bdove/Desktop/LULC Project/CA LULC/sortedselreflulc.gpkg', driver = "GPKG")


#%%


reflulc = gpd.read_file('C:/Users/bdove/Desktop/LULC Project/CA LULC/sortedselreflulc.gpkg')

#%% Colormap

class LULC(MultiValueEnum):
    """Enum class containing basic LULC types"""
    
    NO_DATA = "No Data", 0, "#ffffff"
    GRAIN = "Grain and hay crops", 1, "#ffff00"
    RICE = "Rice", 2, "#054907"
    FIELD_CROPS = "Field Crops", 3, "#ffa500"
    PASTURE = "Pasture", 4, "#806000"
    TRUCK_CROPS = "Truck, nursery, and berry crops", 5, "#069af3"
    DECIDUOUS_FRUITS_NUTS = "Deciduos fruit and nuts", 6, "#95d0fc"
    CITRUS = "Citrus and subtropical", 7, "#967bb6"
    VINEYARDS = "Vineyards", 8, "#dc143c"
    IDLE = "Idle", 9, "#a6a6a6"
    SEMI_AG = "Semi-agricultural and incidental to agriculture", 10, "#8B0000"
    URBAN = "Urban - residential, commerical and industrial, unsegregated", 11, "#FFA07A"
    URBAN_RES = "Urban - residential, single and multi family units, includes trailer parks", 12, "#008080"
    URBAN_COM = "Urban - commerical", 13, "#00FF00"
    URBAN_IND = "Urban - industrial", 14, "#FFB07A"
    URBAN_VAC = "Urban - vacant", 15, "#FF7F50"
    NATIVE_CLASSES = "Native classes, unsegregated", 16, "#FFD700"
    NATIVE_VEG = "Native vegetation", 17, "#FF8C00"
    NATIVE_RIP = "Native riparian vegetation", 18, "#FFFACD"
    WATER = "Water Surface", 19, "#FFEFD5"
    BARREN = "Barren and wasteland", 20, "#BDB76B"
    NOT_SURVEYED = "Not surveyed", 21, "#32CD32"
    ENTRY_DENIED = "Entry Denied", 22, "#006400"
    OUTSIDE_AREA = "Outside area of study", 23, "#00FF7F"
    UNCLASSIFIED_FALLOW = "Unclassified fallow", 24, "#00FA9A"
    YOUNG_PERENNIAL="Young Perennial", 25, "#2E8B57"



    @property
    def id(self):
        return self.values[1]

    @property
    def color(self):
        return self.values[2]


# Reference colormap things
lulc_cmap = ListedColormap([x.color for x in LULC], name="lulc_cmap")
lulc_norm = BoundaryNorm([x - 0.5 for x in range(len(LULC) + 1)], lulc_cmap.N)

land_use_ref_path = 'C:/Users/bdove/Desktop/LULC Project/CA LULC/sortedselreflulc.gpkg'

vector_feature = FeatureType.VECTOR_TIMELESS, "LULC_REFERENCE"

vector_import_task = VectorImportTask(vector_feature, land_use_ref_path)

rasterization_task = VectorToRasterTask(
    vector_feature,
    (FeatureType.MASK_TIMELESS, "LULC"),
    values_column="SYMB_CLASS",
    raster_shape=(FeatureType.MASK, "IS_DATA"),
    raster_dtype=np.uint8,
)

# Define the workflow
workflow_nodes = linearly_connect_tasks(
    add_data, ndvi, ndwi, ndbi, add_sh_validmask, add_valid_count, vector_import_task, rasterization_task, save
)
workflow = EOWorkflow(workflow_nodes)
"""
# Let's visualize it
#workflow.dependency_graph('C:/Users/bdove/Desktop/LULC Project/workflow.jpg')
"""



# %% Generating EOPatches - takes forever try not to run again

# Time interval for the SH request
time_interval = ["2019-01-01", "2019-12-31"]


# Define additional parameters of the workflow
input_node = workflow_nodes[0]
save_node = workflow_nodes[-1]
execution_args = []




for idx, bbox in enumerate(bbox_list[patchIDs]):
    execution_args.append(
        {
            input_node: {"bbox": bbox, "time_interval": time_interval},
            save_node: {"eopatch_folder": f"eopatch_{idx}"},
        }
    )

# Execute the workflow





executor = EOExecutor(workflow, execution_args, save_logs=True)    
executor.run(workers=1)
executor.make_report()


failed_ids = executor.get_failed_executions()
if failed_ids:
    raise RuntimeError(
        f"Execution failed EOPatches with IDs:\n{failed_ids}\n"
        f"For more info check report at {executor.get_report_path()}"
        )
    
    
# %% Loading EOPatches & generating plots



eopatch = EOPatch.load('C:/Users/bdove/Desktop/LULC Project/CA LULC/eopatches/eopatch_0')


fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

date = datetime.datetime(2019, 7, 1)



for i in tqdm(range(len(patchIDs))):
    eopatch_path = os.path.join('C:/Users/bdove/Desktop/LULC Project/CA LULC/eopatches', f"eopatch_{i}")
    eopatch = EOPatch.load(eopatch_path, lazy_loading=True)

    dates = np.array([timestamp.replace(tzinfo=None) for timestamp in eopatch.timestamp])
    closest_date_id = np.argsort(abs(date - dates))[0]

    ax = axs[i // 5][i % 5]
    ax.imshow(np.clip(eopatch.data["BANDS"][closest_date_id][..., [2, 1, 0]] * 3.5, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    del eopatch


fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/sat.png")



# Reference colormap things
lulc_cmap = ListedColormap([x.color for x in LULC], name="lulc_cmap")
lulc_norm = BoundaryNorm([x - 0.5 for x in range(len(LULC) + 1)], lulc_cmap.N)

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 25))

for i in tqdm(range(len(patchIDs))):
    eopatch_path = os.path.join(EOPATCH_FOLDER, f"eopatch_{i}")
    eopatch = EOPatch.load(eopatch_path, lazy_loading=True)

    ax = axs[i // 5][i % 5]
    im = ax.imshow(eopatch.mask_timeless["LULC"].squeeze(), cmap=lulc_cmap, norm=lulc_norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    del eopatch

fig.subplots_adjust(wspace=0, hspace=0)

cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.01, aspect=100)
cb.ax.tick_params(labelsize=20)
cb.set_ticks([entry.id for entry in LULC])
cb.ax.set_xticklabels([entry.name for entry in LULC], rotation=45, fontsize=15)
#plt.show();
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/referencemap.png")



# Calculate min and max counts of valid data per pixel
vmin, vmax = None, None
for i in range(len(patchIDs)):
    eopatch_path = os.path.join(EOPATCH_FOLDER, f"eopatch_{i}")
    eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
    data = eopatch.mask_timeless["VALID_COUNT"].squeeze()
    vmin = np.min(data) if vmin is None else (np.min(data) if np.min(data) < vmin else vmin)
    vmax = np.max(data) if vmax is None else (np.max(data) if np.max(data) > vmax else vmax)

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 25))

for i in tqdm(range(len(patchIDs))):
    eopatch_path = os.path.join(EOPATCH_FOLDER, f"eopatch_{i}")
    eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
    ax = axs[i // 5][i % 5]
    im = ax.imshow(eopatch.mask_timeless["VALID_COUNT"].squeeze(), vmin=vmin, vmax=vmax, cmap=plt.cm.inferno)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    del eopatch

fig.subplots_adjust(wspace=0, hspace=0)

cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.01, aspect=100)
cb.ax.tick_params(labelsize=20)
#plt.show()
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/validpixelcounts.png")




eID = 16
eopatch = EOPatch.load(os.path.join(EOPATCH_FOLDER, f"eopatch_{i}"), lazy_loading=True)

ndvi = eopatch.data["NDVI"]
mask = eopatch.mask["IS_VALID"]
time = np.array(eopatch.timestamp)
t, w, h, _ = ndvi.shape

ndvi_clean = ndvi.copy()
ndvi_clean[~mask] = np.nan  # Set values of invalid pixels to NaN's

# Calculate means, remove NaN's from means
ndvi_mean = np.nanmean(ndvi.reshape(t, w * h), axis=1)
ndvi_mean_clean = np.nanmean(ndvi_clean.reshape(t, w * h), axis=1)
time_clean = time[~np.isnan(ndvi_mean_clean)]
ndvi_mean_clean = ndvi_mean_clean[~np.isnan(ndvi_mean_clean)]

fig = plt.figure(figsize=(20, 5))
plt.plot(time_clean, ndvi_mean_clean, "s-", label="Mean NDVI with cloud cleaning")
plt.plot(time, ndvi_mean, "o-", label="Mean NDVI without cloud cleaning")
plt.xlabel("Time", fontsize=15)
plt.ylabel("Mean NDVI over patch", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(loc=2, prop={"size": 15});
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/cloudcleaningresults.png")



# Calculate temporal mean of NDVI
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 25))

for i in tqdm(range(len(patchIDs))):
    eopatch_path = os.path.join(EOPATCH_FOLDER, f"eopatch_{i}")
    eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
    ndvi = eopatch.data["NDVI"]
    mask = eopatch.mask["IS_VALID"]
    ndvi[~mask] = np.nan
    ndvi_mean = np.nanmean(ndvi, axis=0).squeeze()

    ax = axs[i // 5][i % 5]
    im = ax.imshow(ndvi_mean, vmin=0, vmax=0.8, cmap=plt.get_cmap("YlGn"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    del eopatch

fig.subplots_adjust(wspace=0, hspace=0)

cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.01, aspect=100)
cb.ax.tick_params(labelsize=20)
#plt.show()
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/NDVI.png")



fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 25))

for i in tqdm(range(len(patchIDs))):
    eopatch_path = os.path.join(EOPATCH_FOLDER, f"eopatch_{i}")
    eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
    clp = eopatch.data["CLP"].astype(float) / 255
    mask = eopatch.mask["IS_VALID"]
    clp[~mask] = np.nan
    clp_mean = np.nanmean(clp, axis=0).squeeze()

    ax = axs[i // 5][i % 5]
    im = ax.imshow(clp_mean, vmin=0.0, vmax=0.3, cmap=plt.cm.inferno)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    del eopatch

fig.subplots_adjust(wspace=0, hspace=0)

cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.01, aspect=100)
cb.ax.tick_params(labelsize=20)
#plt.show()
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/cloudprobability.png")


# %% Sampling EO Patches


class ValidDataFractionPredicate:
    """Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid if the
    valid data fraction is above the specified threshold.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold





# LOAD EXISTING EOPATCHES
load = LoadTask(EOPATCH_FOLDER)

# FEATURE CONCATENATION
concatenate = MergeFeatureTask({FeatureType.DATA: ["BANDS", "NDVI", "NDWI", "NDBI"]}, (FeatureType.DATA, "FEATURES"))

# FILTER OUT CLOUDY SCENES
# Keep frames with > 80% valid coverage
valid_data_predicate = ValidDataFractionPredicate(0.8)
filter_task = SimpleFilterTask((FeatureType.MASK, "IS_VALID"), valid_data_predicate)

# LINEAR TEMPORAL INTERPOLATION
# linear interpolation of full time-series and date resampling
resampled_range = ("2019-01-01", "2019-12-31", 15)
linear_interp = LinearInterpolationTask(
    (FeatureType.DATA, "FEATURES"),  # name of field to interpolate
    mask_feature=(FeatureType.MASK, "IS_VALID"),  # mask to be used in interpolation
    copy_features=[(FeatureType.MASK_TIMELESS, "LULC")],  # features to keep
    resample_range=resampled_range,
)

# EROSION
# erode each class of the reference map
erosion = ErosionTask(mask_feature=(FeatureType.MASK_TIMELESS, "LULC", "LULC_ERODED"), disk_radius=1)

# SPATIAL SAMPLING
# Uniformly sample pixels from patches
lulc_type_ids = [lulc_type.id for lulc_type in LULC]

spatial_sampling = FractionSamplingTask(
    features_to_sample=[(FeatureType.DATA, "FEATURES", "FEATURES_SAMPLED"), (FeatureType.MASK_TIMELESS, "LULC_ERODED")],
    sampling_feature=(FeatureType.MASK_TIMELESS, "LULC_ERODED"),
    fraction=0.25,  # a quarter of points
    exclude_values=[0],
)

save = SaveTask(EOPATCH_SAMPLES_FOLDER, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)



# Define the workflow
workflow_nodes = linearly_connect_tasks(load, concatenate, filter_task, linear_interp, erosion, spatial_sampling, save)
workflow = EOWorkflow(workflow_nodes)





execution_args = []
for idx in range(len(patchIDs)):
    execution_args.append(
        {
            workflow_nodes[0]: {"eopatch_folder": f"eopatch_{idx}"},  # load
            workflow_nodes[-2]: {"seed": 42},  # sampling
            workflow_nodes[-1]: {"eopatch_folder": f"eopatch_{idx}"},  # save
        }
    )

executor = EOExecutor(workflow, execution_args, save_logs=True)
executor.run(workers=1)

executor.make_report()

failed_ids = executor.get_failed_executions()
if failed_ids:
    raise RuntimeError(
        f"Execution failed EOPatches with IDs:\n{failed_ids}\n"
        f"For more info check report at {executor.get_report_path()}"
    )
    
# %% Setting up & training model

# Load sampled eopatches
sampled_eopatches = []

for i in range(len(patchIDs)):
    sample_path = os.path.join(EOPATCH_SAMPLES_FOLDER, f"eopatch_{i}")
    sampled_eopatches.append(EOPatch.load(sample_path, lazy_loading=True))


# Definition of the train and test patch IDs, take 80 % for train
test_ID = [0, 8, 16, 19, 20]
test_eopatches = [sampled_eopatches[i] for i in test_ID]
train_ID = [i for i in range(len(patchIDs)) if i not in test_ID]
train_eopatches = [sampled_eopatches[i] for i in train_ID]

# Set the features and the labels for train and test sets
features_train = np.concatenate([eopatch.data["FEATURES_SAMPLED"] for eopatch in train_eopatches], axis=1)
labels_train = np.concatenate([eopatch.mask_timeless["LULC_ERODED"] for eopatch in train_eopatches], axis=0)

features_test = np.concatenate([eopatch.data["FEATURES_SAMPLED"] for eopatch in test_eopatches], axis=1)
labels_test = np.concatenate([eopatch.mask_timeless["LULC_ERODED"] for eopatch in test_eopatches], axis=0)

# Get shape
t, w1, h, f = features_train.shape
t, w2, h, f = features_test.shape


#%%


# Reshape to n x m
features_train = np.moveaxis(features_train, 0, 2).reshape(w1 * h, t * f)
labels_train = labels_train.reshape(w1 * h)
features_test = np.moveaxis(features_test, 0, 2).reshape(w2 * h, t * f)
labels_test = labels_test.reshape(w2 * h)

shapetest = features_train.shape

#%% swapping axis

features_train2 = np.swapaxes(features_test,0,1)
features_test2 = np.swapaxes(features_test,0,1)



# %%


# Set up training classes
labels_unique = np.unique(labels_train)

# Set up the model
model = lgb.LGBMClassifier(
    objective="multiclass", num_class=len(labels_unique), metric="multi_logloss", random_state=42
)

# Train the model
model.fit(features_train, labels_train)



# Save the model
joblib.dump(model, os.path.join(RESULTS_FOLDER, "model_SI_LULC.pkl"))

#%% testing ordinal encoder to fix label bug

from sklearn.preprocessing import OrdinalEncoder

#Create encoder

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

ordinal_encoder.fit(labels_train.reshape(-1,1))
ordinal_encoder.transform(labels_train.reshape(-1,1))

#ordinal_encoder.fit(features_train)
#ordinal_encoder.fit(features_train.reshape(-1,1))


#%%  testing

from sklearn.preprocessing import LabelEncoder
import bisect

le = preprocessing.LabelEncoder()
le.fit(labels_train)

le_classes = np.append(le.classes_, '<unknown>')

#le_dict = dict(zip(le.classes_, le.transform(le.classes_)))



#%% more testing






# %% Predicting test labels

import pandas as pd

# Load the model
model_path = os.path.join(RESULTS_FOLDER, "model_SI_LULC.pkl")
model = joblib.load(model_path)

#features_test = np.swapaxes(features_test,0,1)

#le = preprocessing.LabelEncoder()
#le.fit(labels_test)


predicted_labels_test = model.predict(features_test)

"""
#gives label error, working around
predicted_labels_test = model.predict_proba(features_test)

labeldf = (pd.DataFrame(predicted_labels_test)).T
"""

#%%

import numpy as np


label0 = labeldf[0]
label1 = labeldf[1]

iterlist = []

for i in range(len(labeldf)):
    if label0[i] > label1[i]:
        #ref['Cultivated'] = 0
        iterlist.append(1)
    elif label0[i] < label1[i]:
        #ref['Cultivated'] = 1
        iterlist.append(2)
    elif label0[i] == label1[i]:
        iterlist.append(2)
    else:
        print("error at index ", i)
        
#iterlist = np.array(iterlist)
        
labeldf['label'] = iterlist
        





#%%
class_labels = np.unique(labels_test)
class_names = [lulc_type.name for lulc_type in LULC]
mask = np.in1d(predicted_labels_test, labels_test)
predictions = predicted_labels_test[mask]
true_labels = labels_test[mask]



# Extract and display metrics
f1_scores = metrics.f1_score(true_labels, predictions, labels=class_labels, average=None)
avg_f1_score = metrics.f1_score(true_labels, predictions, average="weighted")
recall = metrics.recall_score(true_labels, predictions, labels=class_labels, average=None)
precision = metrics.precision_score(true_labels, predictions, labels=class_labels, average=None)
accuracy = metrics.accuracy_score(true_labels, predictions)

print("Classification accuracy {:.1f}%".format(100 * accuracy))
print("Classification F1-score {:.1f}%".format(100 * avg_f1_score))
print()
print("             Class              =  F1  | Recall | Precision")
print("         --------------------------------------------------")
for idx, lulctype in enumerate([class_names[idx] for idx in class_labels]):
    line_data = (lulctype, f1_scores[idx] * 100, recall[idx] * 100, precision[idx] * 100)
    print("         * {0:20s} = {1:2.1f} |  {2:2.1f}  | {3:2.1f}".format(*line_data))


# %% Confusion Matrixes


## Confusion Matrixes

# Define the plotting function
def plot_confusion_matrix(
    confusion_matrix,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    ylabel="True label",
    xlabel="Predicted label",
    filename=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2, suppress=True)

    if normalize:
        normalisation_factor = confusion_matrix.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps
        confusion_matrix = confusion_matrix.astype("float") / normalisation_factor

    plt.imshow(confusion_matrix, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = ".2f" if normalize else "d"
    threshold = confusion_matrix.max() / 2.0
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(
            j,
            i,
            format(confusion_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > threshold else "black",
            fontsize=12,
        )

    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)

fig = plt.figure(figsize=(20, 20))

plt.subplot(1, 2, 1)
confusion_matrix_gbm = metrics.confusion_matrix(true_labels, predictions)
plot_confusion_matrix(
    confusion_matrix_gbm,
    classes=[name for idx, name in enumerate(class_names) if idx in class_labels],
    normalize=True,
    ylabel="Truth (LAND COVER)",
    xlabel="Predicted (GBM)",
    title="Confusion matrix",
)

plt.subplot(1, 2, 2)
confusion_matrix_gbm = metrics.confusion_matrix(predictions, true_labels)
plot_confusion_matrix(
    confusion_matrix_gbm,
    classes=[name for idx, name in enumerate(class_names) if idx in class_labels],
    normalize=True,
    xlabel="Truth (LAND COVER)",
    ylabel="Predicted (GBM)",
    title="Transposed Confusion matrix",
)

plt.tight_layout()
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/confusionmatrix.png")






fig = plt.figure(figsize=(20, 5))

label_ids, label_counts = np.unique(labels_train, return_counts=True)

plt.bar(range(len(label_ids)), label_counts)
plt.xticks(range(len(label_ids)), [class_names[i] for i in label_ids], rotation=45, fontsize=20)
plt.yticks(fontsize=20);

# %% Generating ROC Curves - Broken

class_labels = np.unique(np.hstack([labels_test, labels_train]))

scores_test = model.predict_proba(features_test)
labels_binarized = preprocessing.label_binarize(labels_test, classes=class_labels)

fpr, tpr, roc_auc = {}, {}, {}


"""
for idx, lbl in enumerate(class_labels):
    fpr[idx], tpr[idx], _ = metrics.roc_curve(labels_binarized[:, idx], scores_test[:, idx])
    roc_auc[idx] = metrics.auc(fpr[idx], tpr[idx])

plt.figure(figsize=(20, 10))

for idx, lbl in enumerate(class_labels):
    if np.isnan(roc_auc[idx]):
        continue
    plt.plot(
        fpr[idx],
        tpr[idx],
        color=lulc_cmap.colors[lbl],
        lw=2,
        label=class_names[lbl] + " ({:0.5f})".format(roc_auc[idx]),
    )


plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 0.99])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ROC Curve", fontsize=20)
plt.legend(loc="center right", prop={"size": 15})
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/roccurve.png")
"""

# %% Finding most important features

# Feature names
fnames = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "NDBI"]

# Get feature importances and reshape them to dates and features
feature_importances = model.feature_importances_.reshape((t, f))

fig = plt.figure(figsize=(15, 15))
ax = plt.gca()

# Plot the importances
im = ax.imshow(feature_importances, aspect=0.25)
plt.xticks(range(len(fnames)), fnames, rotation=45, fontsize=20)
plt.yticks(range(t), [f"T{i}" for i in range(t)], fontsize=20)
plt.xlabel("Bands and band related features", fontsize=20)
plt.ylabel("Time frames", fontsize=20)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

fig.subplots_adjust(wspace=0, hspace=0)

cb = fig.colorbar(im, ax=[ax], orientation="horizontal", pad=0.01, aspect=100)
cb.ax.tick_params(labelsize=20)
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/importantfeatures.png")

# %% Draw the RGB image

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))

time_id = np.where(feature_importances == np.max(feature_importances))[0][0]

for i in tqdm(range(len(patchIDs))):
    sample_path = os.path.join(EOPATCH_SAMPLES_FOLDER, f"eopatch_{i}")
    eopatch = EOPatch.load(sample_path, lazy_loading=True)
    ax = axs[i // 5][i % 5]
    ax.imshow(np.clip(eopatch.data["FEATURES"][time_id][..., [2, 1, 0]] * 2.5, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    del eopatch

fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/rgbimg.png")

# %% Run patch prediction and export to GEOTiff

class PredictPatchTask(EOTask):
    """
    Task to make model predictions on a patch. Provide the model and the feature,
    and the output names of labels and scores (optional)
    """

    def __init__(self, model, features_feature, predicted_labels_name, predicted_scores_name=None):
        self.model = model
        self.features_feature = features_feature
        self.predicted_labels_name = predicted_labels_name
        self.predicted_scores_name = predicted_scores_name

    def execute(self, eopatch):
        features = eopatch[self.features_feature]

        t, w, h, f = features.shape
        features = np.moveaxis(features, 0, 2).reshape(w * h, t * f)

        predicted_labels = self.model.predict(features)
        predicted_labels = predicted_labels.reshape(w, h)
        predicted_labels = predicted_labels[..., np.newaxis]
        eopatch[(FeatureType.MASK_TIMELESS, self.predicted_labels_name)] = predicted_labels

        if self.predicted_scores_name:
            predicted_scores = self.model.predict_proba(features)
            _, d = predicted_scores.shape
            predicted_scores = predicted_scores.reshape(w, h, d)
            eopatch[(FeatureType.DATA_TIMELESS, self.predicted_scores_name)] = predicted_scores

        return eopatch

# LOAD EXISTING EOPATCHES
load = LoadTask(EOPATCH_SAMPLES_FOLDER)

# PREDICT
predict = PredictPatchTask(model, (FeatureType.DATA, "FEATURES"), "LBL_GBM", "SCR_GBM")

# SAVE
save = SaveTask(EOPATCH_SAMPLES_FOLDER, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)



# EXPORT TIFF
export_tiff = ExportToTiffTask((FeatureType.MASK_TIMELESS, "LBL_GBM"), 'C:/Users/bdove/Desktop/LULC Project/CA LULC/results/predicted_tiff')
tiff_location = os.path.join(RESULTS_FOLDER, "predicted_tiff")
os.makedirs('C:/Users/bdove/Desktop/LULC Project/CA LULC/results/predicted_tiff', exist_ok=True)

workflow_nodes = linearly_connect_tasks(load, predict, export_tiff, save)
workflow = EOWorkflow(workflow_nodes)

test = "/"
# Create a list of execution arguments for each patch
execution_args = []
for i in range(len(patchIDs)):
    execution_args.append(
        {
            workflow_nodes[0]: {"eopatch_folder": f"eopatch_{i}"},
            #workflow_nodes[2]: {"filename": f"{tiff_location}/prediction_eopatch_{i}.tiff"},
            workflow_nodes[2]: {"filename": f"prediction_eopatch_{i}.tiff"},
            workflow_nodes[3]: {"eopatch_folder": f"eopatch_{i}"},
        }
    )

# Run the executor
executor = EOExecutor(workflow, execution_args)
executor.run(workers=1, multiprocess=False)
executor.make_report()

failed_ids = executor.get_failed_executions()
if failed_ids:
    raise RuntimeError(
        f"Execution failed EOPatches with IDs:\n{failed_ids}\n"
        f"For more info check report at {executor.get_report_path()}"
    )
    
# %% Visualizing the prediction

# Reference colormap things
lulc_cmap = ListedColormap([x.color for x in LULC], name="lulc_cmap")
lulc_norm = BoundaryNorm([x - 0.5 for x in range(len(LULC) + 1)], lulc_cmap.N)

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 25))

for i in tqdm(range(len(patchIDs))):
    eopatch_path = os.path.join(EOPATCH_SAMPLES_FOLDER, f"eopatch_{i}")
    eopatch = EOPatch.load(eopatch_path, lazy_loading=True)
    ax = axs[i // 5][i % 5]
    im = ax.imshow(eopatch.mask_timeless["LBL_GBM"].squeeze(), cmap=lulc_cmap, norm=lulc_norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    del eopatch

fig.subplots_adjust(wspace=0, hspace=0)

cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.01, aspect=100)
cb.ax.tick_params(labelsize=20)
cb.set_ticks([entry.id for entry in LULC])
cb.ax.set_xticklabels([entry.name for entry in LULC], rotation=45, fontsize=15)
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/prediction.png")

# %% Visual inspection of patches

# Draw the Reference map
fig = plt.figure(figsize=(20, 20))

idx = np.random.choice(range(len(patchIDs)))
inspect_size = 100

eopatch = EOPatch.load(os.path.join(EOPATCH_SAMPLES_FOLDER, f"eopatch_{idx}"), lazy_loading=True)

w, h = eopatch.mask_timeless["LULC"].squeeze().shape

w_min = np.random.choice(range(w - inspect_size))
w_max = w_min + inspect_size
h_min = np.random.choice(range(h - inspect_size))
h_max = h_min + inspect_size

ax = plt.subplot(2, 2, 1)
plt.imshow(eopatch.mask_timeless["LULC"].squeeze()[w_min:w_max, h_min:h_max], cmap=lulc_cmap, norm=lulc_norm)
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title("Ground Truth", fontsize=20)

ax = plt.subplot(2, 2, 2)
plt.imshow(eopatch.mask_timeless["LBL_GBM"].squeeze()[w_min:w_max, h_min:h_max], cmap=lulc_cmap, norm=lulc_norm)
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title("Prediction", fontsize=20)

ax = plt.subplot(2, 2, 3)
mask = eopatch.mask_timeless["LBL_GBM"].squeeze() != eopatch.mask_timeless["LULC"].squeeze()
plt.imshow(mask[w_min:w_max, h_min:h_max], cmap="gray")
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title("Difference", fontsize=20)

ax = plt.subplot(2, 2, 4)
image = np.clip(eopatch.data["FEATURES"][8][..., [2, 1, 0]] * 3.5, 0, 1)
plt.imshow(image[w_min:w_max, h_min:h_max])
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title("True Color", fontsize=20)

fig.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("C:/Users/bdove/Desktop/LULC Project/CA LULC/finalcomparison.png")


#%% Testing for GeoTiff reading

import rasterio
import rasterio.features
import rasterio.warp

"""
with rasterio.open('C:/Users/bdove/Desktop/LULC Project/CA LULC/results/predicted_tiff/prediction_eopatch_0.tiff') as dataset:
    
    mask = dataset.dataset_mask()
    
    for geom, val in rasterio.features.shapes(
            mask, transform=dataset.transform):
        
        geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4269', geom, precision=6)
"""

dataset = rasterio.open('C:/Users/bdove/Desktop/LULC Project/CA LULC/results/predicted_tiff/prediction_eopatch_0.tiff')