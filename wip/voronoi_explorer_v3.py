from pathlib import Path
import holoviews as hv
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

# base_path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20\naive")
# base_path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20")
# base_path = Path(r"D:\Data\MicroscopyPipeline\ser1\naive")
base_path = Path(r"D:\Data\MicroscopyPipeline\ser1")
path = base_path / "segmented_table.h5"
assert path.exists()


# Consts
magnification_towards_camera = 1
pixel_size_in_microns = 0.345 * magnification_towards_camera
calibration_squared_microns_to_squared_pixel = pixel_size_in_microns**2
typical_cell_area_microns = 400

# Processing methods
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

    return df


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

    return df


def calculate_kymograph(df):
    return df.agg({"area": ["mean", "std"], "perimeter": ["mean", "std"]})


def draw_kymograph(statistic, quantiles):
    # kymograph_df = vor_stats_df.query('area.quantile(.05) < area < area.quantile(.95)').groupby(['timestep', 'bins']).apply(calculate_kymograph)
    kymograph_df = (
        vor_stats_df.query(
            f"area.quantile({quantiles[0]}) < area < area.quantile({quantiles[1]})"
        )
        .groupby(["timestep", "bins"])
        .apply(calculate_kymograph)
    )
    kymograph_df.index.rename(names=["timestep", "bins", "statistic"], inplace=True)

    # pivot = kymograph_df.reset_index().query('statistic == "mean"').pivot(index='timestep', columns='bins', values='area')
    pivot = (
        kymograph_df.reset_index()
        .query(f'statistic == "{statistic}"')
        .pivot(index="timestep", columns="bins", values="area")
    )
    kymo = (
        pivot.sort_index(axis="columns")
        .hvplot.heatmap(flip_yaxis=True)
        .opts(cmap="fire")
    )

    return kymo.opts(width=1000, height=600)


def draw_voronoi_tiling(frame, quantiles):
    # print(quantiles)
    # exit(0)
    phase = h5py.File(base_path / "merged_probabilities.h5").get("data")[
        # phase = h5py.File(base_path / "naive_merge.h5").get("data")[
        # frame,
        # 0,
        # :,
        # :,
        # 0
        frame,
        ...,
    ]
    flourescence = h5py.File(base_path / "merged_probabilities.h5").get("data")[
        # flourescence = h5py.File(base_path / "naive_merge.h5").get("data")[
        #     frame,
        #     0,
        #     :,
        #     :,
        #     0
        frame,
        ...,
    ]

    cells_df = df.query(f"timestep == {frame}").copy()
    cells_df.center_x = cells_df.center_x / phase.shape[1] - 0.5
    cells_df.center_y = -1 * (cells_df.center_y / phase.shape[0] - 0.5)

    center_points = hv.Image(flourescence).opts(cmap="gray") * cells_df.hvplot.scatter(
        x="center_x", y="center_y"
    )

    v_df = df.query(f"timestep == {frame}").copy()

    v_df.center_x = v_df.center_x / phase.shape[1] - 0.5
    v_df.center_y = -1 * (v_df.center_y / phase.shape[0] - 0.5)

    v_df = v_df.groupby("timestep").apply(compute_voronoi).query("valid_region")
    v_df = v_df.groupby("timestep").apply(compute_voronoi_stats)

    # v_df.query('timestep == 0 and area.quantile(.05) < area < area.quantile(.95)')
    # print(f'area.quantile({quantiles[0]}) < area < area.quantile({quantiles[1]})')
    # exit(0)
    # v_df.query(f'area.quantile({quantiles[0]}) < area < area.quantile({quantiles[1]}')

    # voronoi_tiling = hv.Image(phase).opts(cmap='gray') *  hv.Polygons([{('x', 'y'): vert, 'level': Polygon(vert).area} for vert in v_df.query(f'area.quantile({quantiles[0]}) < area < area.quantile({quantiles[1]}').vertices], vdims='level').opts(alpha=0.4, cmap='bky')
    voronoi_tiling = hv.Image(phase).opts(cmap="gray") * hv.Polygons(
        [
            {("x", "y"): vert, "level": Polygon(vert).area}
            for vert in v_df.query(
                f"area.quantile({quantiles[0]}) < area < area.quantile({quantiles[1]})"
            ).vertices
        ],
        vdims="level",
    ).opts(alpha=0.5, cmap="bky")

    return center_points.opts(width=500, height=500) + voronoi_tiling.opts(
        width=500, height=500
    )


df = pd.read_hdf(path, key="table")
valid_regions_df = df.groupby("timestep").apply(compute_voronoi).query("valid_region")
vor_stats_df = valid_regions_df.groupby("timestep").apply(compute_voronoi_stats)


# print(type(df.timestep.max()))


frame_wdgt = pn.widgets.IntSlider(
    name="Frame", start=0, end=int(df.timestep.max()), step=1, value=0
)
statistic_widget = pn.widgets.RadioBoxGroup(
    name="Statistic", options=["mean", "std"], inline=True
)
quantiles_widget = pn.widgets.RangeSlider(
    name="Quantiles", start=0, end=1, value=(0, 0.95), step=0.01
)


# def original(frame, channel):
#     img = stack[frame, 0, :, :, channel]
#     shape = img.shape
#     # img = preprocessing.minmax_scale(img.ravel()).reshape(shape)

#     # return pn.Column(
#     #     hv.plotting.Image(img).opts(colorbar=False, cmap="gray"),
#     #     hv.Histogram(np.histogram(img, 20)),
#     # )
#     return hv.plotting.Image(img).opts(colorbar=False, cmap="gray")


# row = pn.Row(preprocess_image)

# dmap = hv.DynamicMap(original, kdims=["i", "c"])
# bounded_dmap = dmap.redim.values(i=list(range(frames)), c=list(range(channels)))

bounded_dmap_kymograph = pn.bind(
    draw_kymograph,
    statistic=statistic_widget.param.value,
    quantiles=quantiles_widget.param.value_throttled,
)


bounded_dmap_vor_tiling = hv.DynamicMap(
    pn.bind(
        draw_voronoi_tiling,
        frame=frame_wdgt.param.value_throttled,
        quantiles=quantiles_widget.param.value_throttled,
    )
)
# bounded_dmap_kymograph = hv.DynamicMap(
#     pn.bind(
#         draw_kymograph,
#         statistic=statistic_widget.param.value,
#         quantiles=quantiles_widget.param.value_throttled,
#     )
# )

# bounded_dmap_vor_tiling = hv.DynamicMap(
#     pn.bind(
#         draw_voronoi_tiling,
#         frame=frame_wdgt.param.value_throttled,
#         quantiles=quantiles_widget.param.value_throttled,
#     )
# )


# def build_viz(dmap):
#     regridded = regrid(dmap)
#     # regridded = rasterize(dmap)
#     # histogram = regridded.hist(adjoin=False, normed=True)

#     return pn.panel(regridded, sizing_mode="stretch_both")
#     # return (regridded + histogram).cols(1)


bootstrap = pn.template.MaterialTemplate(title="Voronoi Explorer")
bootstrap.sidebar.extend([frame_wdgt, statistic_widget, quantiles_widget])

bootstrap.main.extend([bounded_dmap_vor_tiling, bounded_dmap_kymograph])

bootstrap.show()


# pn.Column(
#     pn.WidgetBox(frame_wdgt, statistic_widget, quantiles_widget),
#     pn.Column(bounded_dmap_vor_tiling, bounded_dmap_kymograph,
#     ),
# ).show()
