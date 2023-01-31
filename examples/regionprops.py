import time
from pathlib import Path
import holoviews as hv
from tqdm.dask import TqdmCallback
import numba
from sklearn import preprocessing
import panel as pn
from holoviews.operation.datashader import regrid
import h5py
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import hvplot.pandas
import skimage
import numpy as np
import trackmate_utils
import utils

hv.extension("bokeh")

# base_data_path = Path(r"data/fucci_60_frames")
base_data_path = Path(r"D:\Data\full_pipeline_tests\left")

red_stack_path = base_data_path / "red.tif"
green_stack_path = base_data_path / "green.tif"

base_model_path = Path(r"../models")
red_segmentation_model = base_model_path / "cellpose/nuclei_red_v2"
green_segmentation_model = base_model_path / "cellpose/nuclei_green_v2"

red_segmentation_map = utils.read_stack(base_data_path / "red_segmented.tiff")
green_segmentation_map = utils.read_stack(base_data_path / "green_segmented.tiff")

# tm_red = trackmate_utils.TrackmateXML(base_data_path / "red_segmented.tiff.xml")
# tm_green = trackmate_utils.TrackmateXML(base_data_path / "green_segmented.tiff.xml")

red_stack = utils.read_stack(base_data_path / "red.tif")
green_stack = utils.read_stack(base_data_path / "green.tif")

# metric = trackmate_utils.CartesianSimilarity(tm_red, tm_green)
# metric_df = metric.calculate_metric_for_all_tracks()
# metric_df.to_hdf(base_data_path / "metric.h5", key="metric")
# metric_df = pd.read_hdf(base_data_path / "metric.h5", key="metric")
# metric = trackmate_utils.CartesianSimilarityFromFile(tm_red, tm_green, metric_df)


def main():
    # from tqdm import tqdm
    #
    # tqdm.pandas()
    #
    red = red_segmentation_map[0, ...]
    green = green_segmentation_map[0, ...]

    red_mask = red != 0
    green_mask = green != 0
    and_mask = np.logical_and(red_mask, green_mask)

    df = pd.DataFrame(
        skimage.measure.regionprops_table(
            red,
            intensity_image=and_mask,
            properties=("label", "centroid", "area", "mean_intensity"),
        )
    )

    input("pause...")


def main_dask():
    pass


if __name__ == "__main__":
    # from dask.distributed import Client
    # import dask.dataframe as dd
    #
    # client = Client(n_workers=32)
    # print(client)

    print("start")
    start = time.perf_counter()
    # main_dask()
    main()
    dask_time = time.perf_counter()
    print("dask done")

    input("enter...")
    print("done")
