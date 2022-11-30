from trackmate_utils import TrackmateXML, make_perimeter, make_path
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import holoviews as hv
import panel as pn
import h5py
from sklearn.preprocessing import MinMaxScaler

hv.extension("bokeh")
pn.extension()


# @cache
# def calculate_intersection(green_track_id, red_track_id, buffer=4):
#     red_track_df = tm_red.trace_track(red_track_id)[0]
#     green_track_df = tm_green.trace_track(green_track_id)[0]
#     min_frame = max(red_track_df.frame.min(), green_track_df.frame.min())
#     max_frame = min(red_track_df.frame.max(), green_track_df.frame.max())
#
#     red_overlap_df = red_track_df.query("@min_frame <= frame <= @max_frame")
#     green_overlap_df = green_track_df.query("@min_frame <= frame <= @max_frame")
#
#     if len(red_overlap_df) < 2 or len(green_overlap_df) < 2:
#         return Point(0, 0)
#
#     red_line = LineString(
#         red_overlap_df[["POSITION_X", "POSITION_Y"]]
#         .astype(float)
#         .itertuples(index=False, name=None)
#     )
#     green_line = LineString(
#         green_overlap_df[["POSITION_X", "POSITION_Y"]]
#         .astype(float)
#         .itertuples(index=False, name=None)
#     )
#
#     return red_line.buffer(buffer).intersection(green_line.buffer(buffer))
#
#
# def calculate_match_index(pairs):
#     return np.array([calculate_intersection(g, r).area for g, r in pairs])


def visualize_spot(images, track_id, frame):
    opts = {
        "aspect": images[0].shape[2] / images[0].shape[1],
        "invert_yaxis": True,
        "responsive": True,
    }
    bounds = (0, 0, images[0].shape[2], images[0].shape[1])

    red_lines = [make_path(tm_red.trace_track(track_id))]
    spot_ids = pd.unique(
        tm_red.tracks.query("TrackID == @track_id")[
            ["SPOT_SOURCE_ID", "SPOT_TARGET_ID"]
        ].values.ravel("K")
    )
    spot = red_spots_df.query("frame == @frame and ID in @spot_ids")

    perimeters = [make_perimeter(spot)]
    #     print(perimeters[0].iloc[0])

    # construct holoviews objects
    hv_images = [
        hv.Image(np.flipud(img[frame, ...]), bounds=bounds).opts(cmap="gray", **opts)
        for img in images
    ]
    cell_x, cell_y = list(
        spot[["POSITION_X", "POSITION_Y"]].itertuples(index=False, name=None)
    )[0]
    #     print(cell_x, cell_y)
    cell_bounds = (cell_x - 10, cell_y - 10, cell_x + 10, cell_y + 10)
    opts_cell = {  # 'height': segmented_image.shape[0],
        "aspect": 1,
        "invert_yaxis": True,
        "responsive": True,
    }
    cell_bounds = (cell_x - 30, cell_y - 30, cell_x + 30, cell_y + 30)
    cell_images = [
        hv.Image(np.flipud(img[frame, ...]), bounds=bounds)[
            cell_x - 30 : cell_x + 30, cell_y - 30 : cell_y + 30
        ].opts(cmap="gray", **opts_cell)
        for img in images
    ]

    cell_perimeter_red = hv.Path([poly.coords for poly in red_lines]).opts(
        line_color="red", line_width=2  # , color="transperant"
    )
    cell_perimeter_green = hv.Polygons(
        [poly.iloc[0].exterior.xy for poly in perimeters]
    ).opts(line_color="red", line_width=3, color=None)
    # cell_perimeter_yellow = hv.Polygons(
    #     [poly.boundary.xy for poly in yellow_polygons if poly]
    # ).opts(line_color="yellow", line_width=3, color="transperant")

    # construct layout
    layout = (
        hv.Layout(
            [
                (img)
                * cell_perimeter_red
                * cell_perimeter_green  # * cell_perimeter_yellow
                for img in hv_images
            ]
        )
        .cols(1)
        .opts(shared_axes=False)
    )
    layout2 = (
        hv.Layout(
            [
                img
                * cell_perimeter_red
                * cell_perimeter_green  # * cell_perimeter_yellow
                for img in cell_images
            ]
        )
        .cols(1)
        .opts(shared_axes=False)
    )
    l1 = [
        (img) * cell_perimeter_red * cell_perimeter_green  # * cell_perimeter_yellow
        for img in hv_images
    ]

    l2 = [
        img * cell_perimeter_red * cell_perimeter_green  # * cell_perimeter_yellow
        for img in cell_images
    ]
    #     return hv.Layout([l1[0], l2[0],l1[1], l2[1],]).cols(2).opts(shared_axes=False)
    return l1, l2


base_path = Path(r"data/fucci_60_frames")

tm_red = TrackmateXML(base_path / "red_segmented.tiff.xml")
red_spots_df = tm_red.spots
red_tracks_df = tm_red.tracks

tm_green = TrackmateXML(base_path / "green_segmented.tiff.xml")
green_spots_df = tm_green.spots
green_tracks_df = tm_green.tracks

# spots in frame 4
green_spots_in_frame_4 = set(green_spots_df.query("frame == 4").index)
red_spots_in_frame_4 = set(red_spots_df.query("frame == 4").index)
# tracks in frame 4
green_tracks_in_frame_4 = (
    green_tracks_df.query(
        "(SPOT_TARGET_ID in @green_spots_in_frame_4 or SPOT_SOURCE_ID in @green_spots_in_frame_4)"
    )
    .TrackID.unique()
    .tolist()
)
red_tracks_in_frame_4 = (
    red_tracks_df.query(
        "(SPOT_TARGET_ID in @red_spots_in_frame_4 or SPOT_SOURCE_ID in @red_spots_in_frame_4)"
    )
    .TrackID.unique()
    .tolist()
)


red_stack = tifffile.TiffFile(base_path / "red.tif").asarray()
green_stack = tifffile.TiffFile(base_path / "green.tif").asarray()


segmented_red = h5py.File(base_path / "red_segmented.h5").get("data")
segmented_green = h5py.File(base_path / "green_segmented.h5").get("data")


scaler = MinMaxScaler()


frame_wdgt = pn.widgets.IntSlider(start=0, end=60)
img_id = pn.widgets.IntSlider(start=1, end=200)
track_id_wdgt = pn.widgets.Select(
    name="track_id", value=15, options=red_tracks_in_frame_4
)


def measure_spot(df):
    masked_array_red = np.ma.masked_where(
        segmented_red[df.frame, ...] != df.image_id, red_stack[df.frame, ...]
    )
    masked_array_green = np.ma.masked_where(
        segmented_red[df.frame, ...] != df.image_id, green_stack[df.frame, ...]
    )
    return (
        masked_array_red.mean(),
        masked_array_red.std(),
        masked_array_green.mean(),
        masked_array_green.std(),
    )


def measure_track(track_id):
    spot_ids = pd.unique(
        tm_red.tracks.query("TrackID == @track_id")[
            ["SPOT_SOURCE_ID", "SPOT_TARGET_ID"]
        ].values.ravel("K")
    )
    spot = red_spots_df.query("ID in @spot_ids").copy()
    spot[["mean_red", "std_red", "mean_green", "std_green"]] = spot.apply(
        measure_spot, axis="columns", result_type="expand"
    )
    return spot


def draw_measurement(track_id):
    df = measure_track(track_id)
    df["std_red"] = df.std_red / df.mean_red
    df["std_green"] = df.std_green / df.mean_green

    df[["mean_red", "mean_green"]] = pd.DataFrame(
        scaler.fit_transform(df[["mean_red", "mean_green"]].values),
        columns=["mean_red", "mean_green"],
        index=df.index,
    )

    df["low_red"] = df["mean_red"] - df["std_red"]
    df["high_red"] = df["mean_red"] + df["std_red"]
    df["low_green"] = df["mean_green"] - df["std_green"]
    df["high_green"] = df["mean_green"] + df["std_green"]

    return (
        df.hvplot(x="frame", y="mean_red").opts(color="red")
        * df.hvplot.area(x="frame", y="low_red", y2="high_red").opts(
            alpha=0.3, color="red"
        )
        * df.hvplot(x="frame", y="mean_green").opts(color="green")
        * df.hvplot.area(x="frame", y="low_green", y2="high_green").opts(
            alpha=0.3, color="green"
        )
    )


@pn.depends(frame_wdgt, track_id_wdgt)
def draw(frame, track_id):
    a, cell = visualize_spot([red_stack, green_stack], track_id, frame)
    gspec = pn.GridSpec(sizing_mode="stretch_width")

    col1 = hv.Layout(a).cols(1)
    col2 = hv.Layout(cell).opts(shared_axes=False).cols(1)
    gspec[0, :3] = col1
    gspec[0, 3:4] = col2
    gspec[1, :] = pn.layout.Divider()
    gspec[2, :] = draw_measurement(track_id) * hv.VLine(frame).opts(
        color="gray", line_width=3
    )
    #     gspec[0, :3] = a[0]
    #     gspec[0, 3:4] = cell[0]
    #     gspec[1, :3] = a[1]
    #     gspec[1, 3:4] = cell[1]
    return gspec


col = pn.Column(frame_wdgt, track_id_wdgt, draw, width=800)
col.show()

# print(hm[hm > 30].nonzero())
# hm_image = hv.Image(
#     hm, bounds=(0, 0, len(red_tracks_in_frame_4), len(green_tracks_in_frame_4))
# )


# In[ ]:


# In[ ]:
