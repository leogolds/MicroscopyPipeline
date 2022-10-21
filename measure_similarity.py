#!/usr/bin/env python
# coding: utf-8

# In[1]:


from trackmate_utils import TrackmateXML
from pathlib import Path
from shapely.geometry.polygon import Polygon
from shapely.geometry.linestring import LineString
from shapely.affinity import translate
from PIL import Image
import numpy as np
import pandas as pd
import tifffile

import holoviews as hv

hv.extension("bokeh")
import panel as pn

pn.extension()


from trackmate_utils import TrackmateXML
from pathlib import Path
import itertools
from functools import cache
from shapely.geometry.polygon import Polygon
from shapely.geometry.linestring import LineString
from shapely.affinity import translate
from PIL import Image
import numpy as np
import pandas as pd
import tifffile
import holoviews as hv
import panel as pn

hv.extension("bokeh")
pn.extension()


def get_track_line(tm, track_id, start, end):
    return tm.trace_track(track_id)[0].query("start <= frame <= end")


@cache
def calculate_similarity(green_track_id, red_track_id):
    red_track_df = tm_red.trace_track(red_track_id)
    green_track_df = tm_green.trace_track(green_track_id)
    min_frame = max(red_track_df.frame.min(), green_track_df.frame.min())
    max_frame = min(red_track_df.frame.max(), green_track_df.frame.max())

    red_track_df.query("@min_frame <= frame <= @max_frame", inplace=True)
    green_track_df.query("@min_frame <= frame <= @max_frame", inplace=True)

    if len(red_track_df) < 2 or len(green_track_df) < 2:
        return np.inf

    # sse_x = ((red_track_df.POSITION_X - green_track_df.POSITION_X) ** 2).sum()
    # sse_y = ((red_track_df.POSITION_Y - green_track_df.POSITION_Y) ** 2).sum()
    sse = (
        (
            (
                red_track_df.reset_index().POSITION_X
                - green_track_df.reset_index().POSITION_X
            )
            ** 2
            + (
                red_track_df.reset_index().POSITION_Y
                - green_track_df.reset_index().POSITION_Y
            )
            ** 2
        )
        ** 0.5
    ).sum()

    # return (sse_x + sse_y) / (max_frame - min_frame)
    return sse / (max_frame - min_frame)


def calculate_match_index(pairs):
    return np.array([calculate_similarity(g, r) for g, r in pairs])


def visualize_spot(images, red_line_id, green_line_id):
    opts = {  # 'height': segmented_image.shape[0],
        "aspect": images[0].shape[1] / images[0].shape[0],
        "invert_yaxis": True,
        "responsive": True,
    }
    bounds = (0, 0, images[0].shape[0], images[0].shape[1])

    # construct holoviews objects
    hv_images = [
        hv.Image(np.flipud(img), bounds=bounds).opts(cmap="gray", **opts)
        for img in images
    ]
    red_line = hv.Path(make_line(tm_red.trace_track(red_line_id)).coords).opts(
        color="red"
    )
    # green_line = hv.Path(make_line(tm_green.trace_track(green_line_id)).coords).opts(
    #     color="green"
    # )

    # construct layout
    layout = (
        hv.Layout([img * red_line for img in hv_images]).cols(1).opts(shared_axes=False)
    )
    return layout


base_path = Path(r"data/fucci_60_frames")
tm_red = TrackmateXML(base_path / "red_segmented.tiff.xml")
tm_green = TrackmateXML(base_path / "green_segmented.tiff.xml")


spots_relevant_columns = [
    "frame",
    "POSITION_X",
    "POSITION_Y",
    "PERIMETER",
    "image_id",
    "AREA",
    "ROI",
]
tracks_relevant_columns = [
    "EDGE_TIME",
    "TrackID",
    "SPOT_SOURCE_ID",
    "SPOT_TARGET_ID",
    "EDGE_X_LOCATION",
    "EDGE_Y_LOCATION",
]

red_spots_df = tm_red.spots[spots_relevant_columns]
green_spots_df = tm_green.spots[spots_relevant_columns]
red_tracks_df = tm_red.tracks[tracks_relevant_columns]
green_tracks_df = tm_green.tracks[tracks_relevant_columns]

# spots in frame 4
green_spots_in_frame_4 = set(green_spots_df.query("frame == 4").index)
red_spots_in_frame_4 = set(red_spots_df.query("frame == 4").index)
# tracks in frame 4
green_tracks_in_frame_4 = green_tracks_df.query(
    "(SPOT_TARGET_ID in @green_spots_in_frame_4 or SPOT_SOURCE_ID in @green_spots_in_frame_4)"
).TrackID.unique()
red_tracks_in_frame_4 = red_tracks_df.query(
    "(SPOT_TARGET_ID in @red_spots_in_frame_4 or SPOT_SOURCE_ID in @red_spots_in_frame_4)"
).TrackID.unique()

combinations = itertools.product(
    green_tracks_in_frame_4.tolist(), red_tracks_in_frame_4.tolist()
)
combinations = list(combinations)


def construct_perimeters(df):
    polygons = df.apply(make_polygon, axis="columns")
    # print(polygons)
    return polygons


def make_polygon(df):
    polygon = Polygon(df.ROI)
    x, y = df.POSITION_X, df.POSITION_Y
    polygon = translate(polygon, x + 0.5, y + 0.5)

    return polygon


def visualize_spot_v3(images, red_track_id, green_track_id, frame):
    opts = {  # 'height': segmented_image.shape[0],
        "aspect": images[0].shape[2] / images[0].shape[1],
        "invert_yaxis": True,
        "responsive": True,
    }
    #     print(images[0].shape)
    bounds = (0, 0, images[0].shape[2], images[0].shape[1])

    red_lines = [make_track(tm_red.trace_track(red_track_id))]
    spot_ids = pd.unique(
        tm_red.tracks.query("TrackID == @red_track_id")[
            ["SPOT_SOURCE_ID", "SPOT_TARGET_ID"]
        ].values.ravel("K")
    )
    red_spot = red_spots_df.query("frame == @frame and ID in @spot_ids")

    green_lines = [make_track(tm_green.trace_track(green_track_id))]
    spot_ids = pd.unique(
        tm_green.tracks.query("TrackID == @green_track_id")[
            ["SPOT_SOURCE_ID", "SPOT_TARGET_ID"]
        ].values.ravel("K")
    )
    green_spot = red_spots_df.query("frame == @frame and ID in @spot_ids")

    perimeters = [
        construct_perimeters(red_spot)
        if not red_spot.empty
        else construct_perimeters(green_spot)
    ]
    #     print(perimeters[0].iloc[0])

    # construct holoviews objects
    hv_images = [
        hv.Image(np.flipud(img[frame, ...]), bounds=bounds).opts(cmap="gray", **opts)
        for img in images
    ]
    cell_x, cell_y = list(
        (
            red_spot[["POSITION_X", "POSITION_Y"]]
            if not red_spot.empty
            else green_spot[["POSITION_X", "POSITION_Y"]]
        ).itertuples(index=False, name=None)
    )[0]
    #     print(cell_x, cell_y)
    #     cell_bounds = (cell_x-10, cell_y-10, cell_x+10, cell_y+10)
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

    cell_path_red = hv.Path([poly.coords for poly in red_lines]).opts(
        line_color="red", line_width=2  # , color="transperant"
    )
    cell_perimeter_red = hv.Polygons(
        [poly.iloc[0].exterior.xy for poly in perimeters]
    ).opts(line_color="red", line_width=3, color="transperant")

    cell_path_green = hv.Path([poly.coords for poly in green_lines]).opts(
        line_color="green", line_width=2  # , color="transperant"
    )
    #     cell_perimeter_green = hv.Polygons(
    #         [poly.iloc[0].exterior.xy for poly in perimeters]
    #     ).opts(line_color="red", line_width=3, color="transperant")
    # cell_perimeter_yellow = hv.Polygons(
    #     [poly.boundary.xy for poly in yellow_polygons if poly]
    # ).opts(line_color="yellow", line_width=3, color="transperant")

    # construct layout
    # layout = (
    #     hv.Layout(
    #         [
    #             (img) * cell_path_red * cell_perimeter_red  # * cell_perimeter_yellow
    #             for img in hv_images
    #         ]
    #     )
    #     .cols(1)
    #     .opts(shared_axes=False)
    # )
    # layout2 = (
    #     hv.Layout(
    #         [
    #             img * cell_path_red * cell_perimeter_red  # * cell_perimeter_yellow
    #             for img in cell_images
    #         ]
    #     )
    #     .cols(1)
    #     .opts(shared_axes=False)
    # )
    l1 = [
        (img)
        * cell_perimeter_red
        * cell_path_red
        * cell_path_green  # * cell_perimeter_yellow
        for img in hv_images
    ]

    l2 = [
        img
        * cell_perimeter_red
        * cell_path_red
        * cell_path_green  # * cell_perimeter_yellow
        for img in cell_images
    ]
    #     return hv.Layout([l1[0], l2[0],l1[1], l2[1],]).cols(2).opts(shared_axes=False)
    return l1, l2


def construct_complex_track(red_track_id, green_track_id):
    df, (overlap_frame_min, overlap_frame_max) = merge_tracks(
        red_track_id, green_track_id
    )

    early_red_frames = df.query("frame < @overlap_frame_min and color == 'red'")
    late_red_frames = df.query("frame > @overlap_frame_max and color == 'red'")
    early_green_frames = df.query("frame < @overlap_frame_min and color == 'green'")
    late_green_frames = df.query("frame > @overlap_frame_max and color == 'green'")

    early_line = make_track(
        early_red_frames if not early_red_frames.empty else early_green_frames
    )
    late_line = make_track(
        late_red_frames if not late_red_frames.empty else late_green_frames
    )
    yellow_line = make_track(df.query("color == 'yellow'"))

    # Returns (red, yellow, green) lines
    return (
        early_line if not early_red_frames.empty else late_line,
        yellow_line,
        late_line if not late_green_frames.empty else early_line,
    )


def merge_tracks(red_track_id, green_track_id):
    red_track_df = tm_red.trace_track(red_track_id)
    green_track_df = tm_green.trace_track(green_track_id)

    overlap_frame_min = max(red_track_df.frame.min(), green_track_df.frame.min())
    overlap_frame_max = min(red_track_df.frame.max(), green_track_df.frame.max())

    overlap_red_frames = red_track_df.query(
        "@overlap_frame_min <= frame <= @overlap_frame_max+1"
    )
    overlap_green_frames = green_track_df.query(
        "@overlap_frame_min <= frame <= @overlap_frame_max+1"
    )
    yellow_frames = (
        pd.concat([overlap_red_frames, overlap_green_frames]).groupby(level=0).mean()
    )

    red_frames = red_track_df.query(
        "frame < @overlap_frame_min or frame > @overlap_frame_max"
    )
    green_frames = green_track_df.query(
        "frame < @overlap_frame_min or frame > @overlap_frame_max"
    )

    yellow_frames["color"] = "yellow"
    red_frames["color"] = "red"
    green_frames["color"] = "green"

    return pd.concat([red_frames, green_frames, yellow_frames]).reset_index(), (
        overlap_frame_min,
        overlap_frame_max,
    )


def visualize_spot_v4(images, red_track_id, green_track_id, frame):
    opts = {  # 'height': segmented_image.shape[0],
        "aspect": images[0].shape[2] / images[0].shape[1],
        "invert_yaxis": True,
        "responsive": True,
    }
    #     print(images[0].shape)
    bounds = (0, 0, images[0].shape[2], images[0].shape[1])

    red_line, yellow_line, green_line = construct_complex_track(
        red_track_id, green_track_id
    )

    red_lines = [red_line]
    spot_ids = pd.unique(
        tm_red.tracks.query("TrackID == @red_track_id")[
            ["SPOT_SOURCE_ID", "SPOT_TARGET_ID"]
        ].values.ravel("K")
    )
    red_spot = red_spots_df.query("frame == @frame and ID in @spot_ids")

    green_lines = [green_line]
    spot_ids = pd.unique(
        tm_green.tracks.query("TrackID == @green_track_id")[
            ["SPOT_SOURCE_ID", "SPOT_TARGET_ID"]
        ].values.ravel("K")
    )
    green_spot = red_spots_df.query("frame == @frame and ID in @spot_ids")

    yellow_lines = [yellow_line]

    perimeters = [
        construct_perimeters(red_spot)
        if not red_spot.empty
        else construct_perimeters(green_spot)
    ]
    #     print(perimeters[0].iloc[0])

    # construct holoviews objects
    hv_images = [
        hv.Image(np.flipud(img[frame, ...]), bounds=bounds).opts(cmap="gray", **opts)
        for img in images
    ]
    cell_x, cell_y = list(
        (
            red_spot[["POSITION_X", "POSITION_Y"]]
            if not red_spot.empty
            else green_spot[["POSITION_X", "POSITION_Y"]]
        ).itertuples(index=False, name=None)
    )[0]
    #     print(cell_x, cell_y)
    #     cell_bounds = (cell_x-10, cell_y-10, cell_x+10, cell_y+10)
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

    cell_path_red = hv.Path(
        [poly.coords for poly in red_lines if not poly.is_empty]
    ).opts(
        line_color="red", line_width=2  # , color="transperant"
    )
    cell_perimeter_red = hv.Polygons(
        [poly.iloc[0].exterior.xy for poly in perimeters]
    ).opts(line_color="red", line_width=3, color="transperant")

    cell_path_yellow = hv.Path([poly.coords for poly in yellow_lines]).opts(
        line_color="yellow", line_width=3  # , color="transperant"
    )
    cell_path_green = hv.Path(
        [poly.coords for poly in green_lines if not poly.is_empty]
    ).opts(
        line_color="green", line_width=3  # , color="transperant"
    )
    #     cell_perimeter_green = hv.Polygons(
    #         [poly.iloc[0].exterior.xy for poly in perimeters]
    #     ).opts(line_color="red", line_width=3, color="transperant")
    # cell_perimeter_yellow = hv.Polygons(
    #     [poly.boundary.xy for poly in yellow_polygons if poly]
    # ).opts(line_color="yellow", line_width=3, color="transperant")

    # construct layout
    # layout = (
    #     hv.Layout(
    #         [
    #             (img) * cell_path_red * cell_perimeter_red  # * cell_perimeter_yellow
    #             for img in hv_images
    #         ]
    #     )
    #     .cols(1)
    #     .opts(shared_axes=False)
    # )
    # layout2 = (
    #     hv.Layout(
    #         [
    #             img * cell_path_red * cell_perimeter_red  # * cell_perimeter_yellow
    #             for img in cell_images
    #         ]
    #     )
    #     .cols(1)
    #     .opts(shared_axes=False)
    # )
    l1 = [
        (img)
        * cell_perimeter_red
        * cell_path_red
        * cell_path_green
        * cell_path_yellow  # * cell_perimeter_yellow
        for img in hv_images
    ]

    l2 = [
        img
        * cell_perimeter_red
        * cell_path_red
        * cell_path_green
        * cell_path_yellow  # * cell_perimeter_yellow
        for img in cell_images
    ]
    #     return hv.Layout([l1[0], l2[0],l1[1], l2[1],]).cols(2).opts(shared_axes=False)
    return l1, l2


#     return (layout+layout2).cols(2).opts(shared_axes=False)
#     return layout, layout2


def make_track(df):
    line = LineString(
        df[["POSITION_X", "POSITION_Y"]]
        .astype(float)
        .itertuples(index=False, name=None)
    )
    return line


frame_wdgt = pn.widgets.IntSlider(start=0, end=tm_red.spots.frame.max().item())
red_id = pn.widgets.Select(
    name="red_id", options=tm_red.tracks.TrackID.unique().tolist()
)
green_id = pn.widgets.Select(
    name="green_id", options=tm_red.tracks.TrackID.unique().tolist()
)
green_id.value = 5
red_id.value = 7

red_stack = tifffile.TiffFile(base_path / "red.tif").asarray()
green_stack = tifffile.TiffFile(base_path / "green.tif").asarray()


@pn.depends(frame_wdgt, red_id, green_id)
def draw(frame, red_track, green_track):
    # a, cell = visualize_spot_v3(
    a, cell = visualize_spot_v4(
        [red_stack, green_stack],
        red_track,
        green_track,
        frame,
    )
    gspec = pn.GridSpec(sizing_mode="stretch_width")

    col1 = hv.Layout(a).cols(1)
    col2 = hv.Layout(cell).opts(shared_axes=False).cols(1)
    gspec[0, :3] = col1
    gspec[0, 3:4] = col2
    gspec[1, :] = pn.layout.Divider()
    # gspec[2, :] = draw_measurement(track_id) * hv.VLine(frame).opts(
    #     color="gray", line_width=3
    # )
    #     gspec[0, :3] = a[0]
    #     gspec[0, 3:4] = cell[0]
    #     gspec[1, :3] = a[1]
    #     gspec[1, 3:4] = cell[1]
    return gspec


col = pn.Column(frame_wdgt, red_id, green_id, draw, width=800)
col.show()

# # In[6]:
#
# # calculate_similarity(*combinations[104])
# calculate_similarity(5, 7)
# hm = calculate_match_index(combinations).reshape(
#     (len(green_tracks_in_frame_4), len(red_tracks_in_frame_4))
# )


# def make_line(df):
#     line = LineString(
#         # df[["POSITION_X", "POSITION_Y"]]
#         df[["POSITION_Y", "POSITION_X"]]
#         .astype(float)
#         .itertuples(index=False, name=None)
#     )
#     return line
#
#
# high_match_tracks = np.transpose((hm > 900).nonzero())
# perimeters = [
#     (
#         make_line(tm_red.trace_track(r)).buffer(4),
#         make_line(tm_green.trace_track(g)).buffer(4),
#     )
#     for g, r in [
#         (25, 5),
#     ]
# ]
# red_perim, green_perim = list(zip(*perimeters)) if perimeters else [], []
# # red_perim = [poly for poly in red_perim if isinstance(poly, LineString)]
# # green_perim = [poly for poly in green_perim if isinstance(poly, LineString)]
# merged_perimeters = [calculate_similarity(r, g) for r, g in high_match_tracks]
#
#
# red_stack = tifffile.TiffFile(base_path / "red.tif").asarray()
# green_stack = tifffile.TiffFile(base_path / "green.tif").asarray()
#
# frame_wdgt = pn.widgets.IntSlider(start=0, end=20)
# img_id = pn.widgets.IntSlider(start=1, end=200)
# red_track_wdgt = pn.widgets.Select(options=red_tracks_in_frame_4.tolist())
# green_track_wdgt = pn.widgets.Select(options=green_tracks_in_frame_4.tolist())
#
#
# @pn.depends(frame_wdgt, red_track_wdgt, green_track_wdgt)
# def draw(frame, red_track, green_track):
#     return visualize_spot(
#         [red_stack[frame, ...], green_stack[frame, ...]],
#         red_track,
#         green_track,
#     )
#
#
# col = pn.Column(frame_wdgt, img_id, red_track_wdgt, green_track_wdgt, draw, width=800)
# col.show()
#
# # # print(hm[hm > 30].nonzero())
# # # hm_image = hv.Image(
# # #     hm, bounds=(0, 0, len(red_tracks_in_frame_4), len(green_tracks_in_frame_4))
# # # )
# # # pn.Row(hm_image).show()
#
#
# # In[ ]:
#
#
# green_perim[1]
