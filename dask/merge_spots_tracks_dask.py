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

import trackmate_utils
import utils

hv.extension("bokeh")

base_data_path = Path(r"../data/fucci_60_frames")
red_stack_path = base_data_path / "red.tif"
green_stack_path = base_data_path / "green.tif"

base_model_path = Path(r"../models")
red_segmentation_model = base_model_path / "cellpose/nuclei_red_v2"
green_segmentation_model = base_model_path / "cellpose/nuclei_green_v2"

red_segmentation_map = utils.read_stack(base_data_path / "red_segmented.tiff")
green_segmentation_map = utils.read_stack(base_data_path / "green_segmented.tiff")

tm_red = trackmate_utils.TrackmateXML(base_data_path / "red_segmented.tiff.xml")
tm_green = trackmate_utils.TrackmateXML(base_data_path / "green_segmented.tiff.xml")

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
    # tracks = tm_red.tracks.reset_index(drop=True)
    # tracks.to_hdf("bla.h5", key="tracks", format="table", mode="w")
    # tm_red.spots.drop(["ROI", "roi_polygon"], axis="columns").to_hdf(
    #     "bla.h5", key="spots", format="table", mode="a"
    # )
    # exit()

    tracks_dd = dd.read_hdf("bla.h5", key="tracks", chunksize=1000).persist()
    spots_dd = dd.read_hdf("bla.h5", key="spots", chunksize=1000).persist()
    # tracks_dd = pd.read_hdf("bla.h5", key="tracks")
    # spots_dd = pd.read_hdf("bla.h5", key="spots")
    # input("pause...")

    merged_1 = spots_dd.merge(
        tracks_dd[["TrackID", "SPOT_SOURCE_ID"]],
        how="left",
        left_index=True,
        right_on="SPOT_SOURCE_ID",
        indicator=True,
    ).persist()
    merged_2 = (
        merged_1.query('_merge != "both"')
        .drop(["TrackID", "_merge", "SPOT_SOURCE_ID"], axis="columns")
        .merge(
            tracks_dd[["TrackID", "SPOT_TARGET_ID"]],
            how="left",
            left_index=True,
            right_on="SPOT_TARGET_ID",
            indicator=True,
        )
        .drop(["SPOT_TARGET_ID", "_merge"], axis="columns")
        .persist()
    )
    answer = dd.concat(
        [
            merged_1.query('_merge == "both"').drop(
                ["_merge", "SPOT_SOURCE_ID"], axis="columns"
            ),
            merged_2,
        ]
    ).compute()
    input("pause...")


def main_dask():
    # df = dd.read_hdf(path, key="table")
    # df = dd.read_hdf(path, key="table", chunksize=10000)
    df = dd.read_hdf(path, key="table", sorted_index=True, chunksize=100000)
    meta_valid_regions = pd.DataFrame(
        columns=[
            "center_y",
            "center_x",
            "timestep",
            "vertice_ids",
            "valid_region",
            "vertices",
        ]
    )
    valid_regions_schema = {
        "center_y": np.float64,
        "center_x": np.float64,
        "timestep": np.int64,
        "vertice_ids": object,
        "valid_region": bool,
        "vertices": object,
    }
    vor_stats_schema = {
        "center_y": np.float64,
        "center_x": np.float64,
        "timestep": np.int64,
        "vertice_ids": object,
        "valid_region": bool,
        "vertices": object,
        "area": np.float64,
        "perimeter": np.float64,
        "bins": np.int64,
    }
    meta_valid_regions = meta_valid_regions.astype(valid_regions_schema)
    valid_regions_df = (
        df.groupby("timestep")
        .apply(compute_voronoi, meta=meta_valid_regions)
        .query("valid_region")
        .persist()
        # .compute()
    )
    vor_stats_df = (
        valid_regions_df.groupby("timestep")
        .apply(compute_voronoi_stats, meta=vor_stats_schema)
        .compute()
    )

    return vor_stats_df


if __name__ == "__main__":
    from dask.distributed import Client
    import dask.dataframe as dd

    client = Client(n_workers=32)
    print(client)

    print("start")
    start = time.perf_counter()
    # main_dask()
    main()
    dask_time = time.perf_counter()
    print("dask done")

    input("enter...")
    print("done")
