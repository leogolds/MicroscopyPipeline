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

from trackmate_utils import TrackmateXML
from pathlib import Path
import itertools
from functools import cache
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
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


@cache
def calculate_intersection(green_track_id, red_track_id, buffer=4):
    red_track_df = tm_red.trace_track(red_track_id)
    green_track_df = tm_green.trace_track(green_track_id)
    min_frame = max(red_track_df.frame.min(), green_track_df.frame.min())
    max_frame = min(red_track_df.frame.max(), green_track_df.frame.max())

    red_overlap_df = red_track_df.query("@min_frame <= frame <= @max_frame")
    green_overlap_df = green_track_df.query("@min_frame <= frame <= @max_frame")

    if len(red_overlap_df) < 2 or len(green_overlap_df) < 2:
        return Point(0, 0)

    red_line = LineString(
        red_overlap_df[["POSITION_X", "POSITION_Y"]]
        .astype(float)
        .itertuples(index=False, name=None)
    )
    green_line = LineString(
        green_overlap_df[["POSITION_X", "POSITION_Y"]]
        .astype(float)
        .itertuples(index=False, name=None)
    )

    return red_line.buffer(buffer).intersection(green_line.buffer(buffer))


def calculate_match_index(pairs):
    return np.array([calculate_intersection(g, r).area for g, r in pairs])


def visualize_spot(images, red_polygons, green_polygons, yellow_polygons):
    opts = {  # 'height': segmented_image.shape[0],
        "aspect": images[0].shape[1] / images[0].shape[0],
        "invert_yaxis": True,
        "responsive": True,
    }
    bounds = (0, 0, images[0].shape[1], images[0].shape[0])

    # construct holoviews objects
    hv_images = [
        hv.Image(np.flipud(img), bounds=bounds).opts(cmap="gray", **opts)
        for img in images
    ]
    cell_perimeter_red = hv.Path([poly.coords for poly in red_polygons]).opts(
        line_color="red", line_width=3, color="transperant"
    )
    cell_perimeter_green = hv.Path([poly.coords for poly in green_polygons]).opts(
        line_color="green", line_width=3, color="transperant"
    )
    cell_perimeter_yellow = hv.Polygons(
        [poly.boundary.xy for poly in yellow_polygons if poly]
    ).opts(line_color="yellow", line_width=3, color="transperant")

    # construct layout
    layout = (
        hv.Layout(
            [
                img * cell_perimeter_red * cell_perimeter_green * cell_perimeter_yellow
                for img in hv_images
            ]
        )
        .cols(1)
        .opts(shared_axes=False)
    )
    return layout


base_path = Path(r"D:\Data\MicroscopyPipeline\ser1")
tm_red = TrackmateXML(base_path / "segmented_red.xml")
tm_green = TrackmateXML(base_path / "segmented_green.xml")


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

combinations = itertools.product(green_tracks_in_frame_4, red_tracks_in_frame_4)
combinations = list(combinations)

hm = calculate_match_index(combinations).reshape(
    (len(green_tracks_in_frame_4), len(red_tracks_in_frame_4))
)


def make_track(df):
    line = LineString(
        df[["POSITION_X", "POSITION_Y"]]
        .astype(float)
        .itertuples(index=False, name=None)
    )
    return line


# high_match_tracks = np.transpose((hm > 150).nonzero())
high_match_tracks = (hm > 300).nonzero()
# perimeters = [
#     (
#         make_track(tm_red.trace_track(red_tracks_in_frame_4[r])[0]).buffer(4).boundary,
#         make_track(tm_green.trace_track(green_tracks_in_frame_4[g])[0])
#         .buffer(4)
#         .boundary,
#     )
#     for g, r in high_match_tracks
# ]
# red_perim, green_perim = list(zip(*perimeters))
# red_perim = [line.buffer(4) for line in red_perim if isinstance(line, LineString)]
# green_perim = [line.buffer(4) for line in green_perim if isinstance(line, LineString)]
red_perim = [
    make_track(tm_red.trace_track(red_tracks_in_frame_4[track_index]))
    for track_index in high_match_tracks[1]
]
green_perim = [
    make_track(tm_green.trace_track(green_tracks_in_frame_4[track_index]))
    for track_index in high_match_tracks[0]
]
merged_perimeters = [
    calculate_intersection(g, r) for g, r in np.transpose(high_match_tracks)
]
merged_perimeters = [poly for poly in merged_perimeters if poly]


red_stack = tifffile.TiffFile(base_path / "red_contrast_short_cropped.tif").asarray()
green_stack = tifffile.TiffFile(
    base_path / "green_contrast_short_cropped.tif"
).asarray()

frame_wdgt = pn.widgets.IntSlider(start=0, end=20)
img_id = pn.widgets.IntSlider(start=1, end=200)


@pn.depends(frame_wdgt)
def draw(frame):
    return visualize_spot(
        [red_stack[frame, ...], green_stack[frame, ...]],
        [],
        [],
        # red_perim,
        # green_perim,
        merged_perimeters,
    )


col = pn.Column(frame_wdgt, img_id, draw, width=800)
col.show()

# print(hm[hm > 30].nonzero())
# hm_image = hv.Image(
#     hm, bounds=(0, 0, len(red_tracks_in_frame_4), len(green_tracks_in_frame_4))
# )
# pn.Row(hm_image).show()
