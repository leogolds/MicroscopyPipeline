import time
from pathlib import Path
import holoviews as hv
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


hv.extension("bokeh")

base_path = Path(r"D:\Data\MicroscopyPipeline\ser1")
path = base_path / "segmented_table.h5"
assert path.exists()
# df = pd.read_hdf(path, key="table").reset_index(drop=True)
# df.to_hdf(path, key="table", format="table")
# exit()

# Consts
magnification_towards_camera = 1
# pixel_size_in_microns = 0.345 * magnification_towards_camera
pixel_size_in_microns = 1 * magnification_towards_camera
calibration_squared_microns_to_squared_pixel = pixel_size_in_microns**2

global_df = pd.DataFrame(
    columns=["center_x", "center_y", "vertice_ids", "valid_region", "vertices"]
)
global_df_list = []


def compute_voronoi(df):
    # print(df.head())
    vor = Voronoi(df[["center_x", "center_y"]])
    # = pd.DataFrame()

    df["vertice_ids"] = [vor.regions[i] for i in vor.point_region]
    df["valid_region"] = [True if min(l) != -1 else False for l in df.vertice_ids]
    df["vertices"] = [
        np.array([vor.vertices[vertice_id] for vertice_id in vertice_ids])
        for vertice_ids in df.vertice_ids
    ]

    global global_df_list
    # global_df = pd.concat([global_df, df])
    # global_df_list.append(df)
    return df


global_stats_list = []


def compute_voronoi_stats(df):
    df["area"] = [
        Polygon(vert).area * calibration_squared_microns_to_squared_pixel
        for vert in df.vertices
    ]
    df["perimeter"] = [
        Polygon(vert).length * pixel_size_in_microns for vert in df.vertices
    ]

    horizontal_bins = range(0, 4096, 40)
    df["bins"] = pd.cut(
        df.center_x, bins=horizontal_bins, labels=range(len(horizontal_bins) - 1)
    )

    # global_stats_list.append(df)
    return df


def main():
    # from tqdm import tqdm
    #
    # tqdm.pandas()
    #
    # global global_df_list
    # global global_stats_list

    df = pd.read_hdf(path, key="table")
    # df_4 = df.query("timestep < 4").copy()
    # df = df.query("timestep < 80").copy()
    # print(f"rows: {len(df)}")
    # df_400 = df.query("timestep < 400").copy()
    # start = time.perf_counter()

    # df = df.groupby("timestep").progress_apply(compute_voronoi)
    df = df.groupby("timestep").apply(compute_voronoi)
    # apply = time.perf_counter()
    # print(f"appply took: {apply-start}")
    # concat_df = pd.concat(global_df_list, ignore_index=True)
    # concat = time.perf_counter()
    # print(f"concat took: {concat - apply}")
    # print(f"len orig {len(df)}, applied {len(concat_df)}")
    #
    # valid_regions_df = concat_df.query("valid_region")
    # print(f"query took: {time.perf_counter()-apply}")
    valid_regions_df = (
        # df.groupby("timestep").progress_apply(compute_voronoi).query("valid_region")
        df.groupby("timestep")
        .apply(compute_voronoi)
        .query("valid_region")
    )
    # # valid_regions_df = (
    # #     df_400.groupby("timestep").progress_apply(compute_voronoi).query("valid_region")
    # # )

    # vor_stats_df = valid_regions_df.groupby("timestep").progress_apply(
    vor_stats_df = valid_regions_df.groupby("timestep").apply(compute_voronoi_stats)
    # result = pd.concat(global_stats_list)
    #
    # global_df_list = []
    # global_stats_list = []

    return vor_stats_df


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
    main_dask()
    dask_time = time.perf_counter()
    print("dask done")
    # main()
    pandas_time = time.perf_counter()

    print(f"dask time: {dask_time-start}")
    print(f"pandas time: {pandas_time-dask_time}")

    input("enter...")
    print("done")
