"""
TrackmateXML
For reading .xml files generated by ImageJ TrackMate https://imagej.net/TrackMate
v1.0
(c) R.Harkes - NKI

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import itertools
from enum import Enum, auto
from typing import Iterable

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

import param
import tqdm
from shapely.affinity import translate
from shapely.geometry import Polygon, LineString
from shapely.geometry.linestring import LineString
from functools import cache
import holoviews as hv
import hvplot.pandas
from sklearn.preprocessing import MinMaxScaler

from utils import view_stacks

hv.extension("bokeh")
import panel as pn


def pairwise_iterator(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


spots_relevant_columns = [
    "frame",
    "POSITION_X",
    "POSITION_Y",
    "PERIMETER",
    "image_id",
    "AREA",
    "ROI",
    "roi_polygon",
]
tracks_relevant_columns = [
    "EDGE_TIME",
    "TrackID",
    "SPOT_SOURCE_ID",
    "SPOT_TARGET_ID",
    "EDGE_X_LOCATION",
    "EDGE_Y_LOCATION",
]


class TrackmateXML:
    """
    Derived from https://github.com/rharkes/pyTrackMateXML/blob/master/trackmatexml.py and updated with custom features
    Trackmate-xml is a class around trackmate xml files to simplify some typical operations on the files, while still
    maintaining access to the raw data.
    """

    class_version = 1.0

    def __init__(self, filename):
        if isinstance(filename, str):
            self.pth = Path(filename)
        elif isinstance(filename, Path):
            self.pth = filename
        else:
            raise ValueError("not a valid filename")

        if self.pth.suffix == ".h5":
            store = pd.HDFStore(self.pth)
            self.spots = store.spots
            self.tracks = store.tracks
            self.filteredtracks = store.filtered_tracks
            other_info = store.other_info
            self.version = other_info.version[0]
            store.close()
        elif self.pth.suffix == ".xml":
            etree = ET.parse(self.pth)
            root = etree.getroot()
            if not root.tag == "TrackMate":
                raise ValueError("Not a TrackmateXML")
            self.version = root.attrib["version"]
            self.spots = self.__loadspots(root)
            self.tracks = self.__loadtracks(root)
            self.filteredtracks = self.__loadfilteredtracks(root)
        else:
            raise ValueError("{0} is not avalid file suffix".format(self.pth.suffix))

    def save(self, filename, create_new=True):
        """
        Saves the spots, tracks and filteredtracks to an HDFStore
        """
        if isinstance(filename, str):
            pth = Path(filename)
        elif isinstance(filename, Path):
            pth = filename
        else:
            raise ValueError("not a valid filename")
        if pth.exists() & create_new:
            pth.unlink()
        store = pd.HDFStore(pth)
        store["spots"] = self.spots
        store["tracks"] = self.tracks
        store["filtered_tracks"] = self.filteredtracks
        other_info = {
            "version": self.version,
            "class_version": TrackmateXML.class_version,
        }
        store["other_info"] = pd.DataFrame(other_info, index=[0])
        store.close()

    @staticmethod
    def __loadfilteredtracks(root):
        """
        Loads all filtered tracks from xml
        :param root: root of xml
        :return: filtered tracks
        """
        filtered_tracks = []
        for track in root.iter("TrackID"):
            track_values = track.attrib
            track_values["TRACK_ID"] = int(track_values.pop("TRACK_ID"))
            filtered_tracks.append(track_values)
        ftracks = pd.DataFrame(filtered_tracks)
        return ftracks

    @staticmethod
    def __loadtracks(root):
        """
        load all tracks in the .xml file
        :param root: root of .xml file
        :return: tracks as pandas dataframe
        """
        all_tracks = []
        for track in root.iter("Track"):
            curr_track = int(track.attrib["TRACK_ID"])
            all_edges = []
            for edge in track:
                edge_values = edge.attrib
                edge_values["SPOT_SOURCE_ID"] = int(edge_values.pop("SPOT_SOURCE_ID"))
                edge_values["SPOT_TARGET_ID"] = int(edge_values.pop("SPOT_TARGET_ID"))
                edge_values["TrackID"] = curr_track
                all_edges.append(edge_values)
            all_tracks.append(pd.DataFrame(all_edges))
        tracks = pd.concat(all_tracks)
        # return tracks
        # TODO align track and spots ID field usage
        return tracks[tracks_relevant_columns]

    @staticmethod
    def __loadspots(root):
        """
        Loads all spots in the xml file
        :return: spots as pandas dataframe
        """
        # load all spots
        all_frames = []
        for spots_in_frame in root.iter("SpotsInFrame"):
            curr_frame = spots_in_frame.attrib["frame"]
            # go over all spots in the frame
            all_spots = []
            for spot in spots_in_frame:
                spot_values = spot.attrib
                spot_values.pop("name")  # not needed
                spot_values["frame"] = int(curr_frame)
                spot_values["ID"] = int(
                    spot_values.pop("ID")
                )  # we want ID to be integer, so we can index later
                spot_values["POSITION_X"] = float(spot_values.pop("POSITION_X"))
                spot_values["POSITION_Y"] = float(spot_values.pop("POSITION_Y"))
                spot_values["image_id"] = int(
                    float(spot_values.get("MAX_INTENSITY_CH1"))
                )
                spot_values["ROI"] = [
                    (float(x), float(y))
                    for x, y in pairwise_iterator(spot.text.split(" "))
                ]
                all_spots.append(spot_values)
            all_frames.append(pd.DataFrame(all_spots))

        spots = pd.concat(all_frames)
        spots.set_index("ID", inplace=True, verify_integrity=True)
        # spots = spots.astype("float")

        spots["roi_polygon"] = spots.apply(make_polygon, axis="columns")
        spots["AREA"] = pd.to_numeric(spots.AREA)

        # return spots
        return spots[spots_relevant_columns]

    @cache
    def trace_track(self, track_id, verbose=False):
        """
        Traces a track over all spots.
        :param verbose: report if a split is found
        :param track_id:
        """
        # assert isinstance(track_id, int)
        # Tracks consist of edges. The edges are not sorted
        current_track = self.tracks[self.tracks["TrackID"] == track_id]
        if current_track.empty:
            raise ValueError("track {0} not found".format(track_id))
        track_splits = []
        source_spots = self.spots.loc[
            current_track["SPOT_SOURCE_ID"].values
        ].reset_index()
        target_spots = self.spots.loc[
            current_track["SPOT_TARGET_ID"].values
        ].reset_index()
        currentindex = source_spots["frame"].idxmin()
        whole_track = [source_spots.loc[currentindex], target_spots.loc[currentindex]]
        # can we continue from the target to a new source?
        current_id = target_spots["ID"].loc[currentindex]
        currentindex = source_spots.index[source_spots["ID"] == current_id].tolist()
        while len(currentindex) > 0:
            if len(currentindex) > 1:
                currentindex = currentindex[0]
                fr = target_spots["frame"].loc[currentindex]
                if verbose:
                    print(
                        "Got a split at frame {0} Will continue on branch 0".format(
                            int(fr)
                        )
                    )
                    # but so far we do nothing with this knowledge
                track_splits.append(fr)
            else:
                currentindex = currentindex[0]
            whole_track.append(target_spots.loc[currentindex])
            current_id = target_spots["ID"].loc[currentindex]
            currentindex = source_spots.index[source_spots["ID"] == current_id].tolist()
        whole_track = pd.concat(whole_track, axis=1).T.reset_index(drop=True)

        # line = LineString(
        #     whole_track[["POSITION_X", "POSITION_Y"]]
        #     .astype(float)
        #     .itertuples(index=False, name=None)
        # )
        return whole_track  # , track_splits
        # return line, whole_track, track_splits


def make_perimeter(df):
    return df.apply(make_polygon, axis="columns")


def make_hv_perimeter(df):
    df = df.apply(make_polygon, axis="columns")
    return hv.Polygons(df.iloc[0].exterior.xy)


def make_polygon(df):
    polygon = Polygon(df.ROI)
    x, y = df.POSITION_X, df.POSITION_Y
    polygon = translate(polygon, x + 0.5, y + 0.5)

    return polygon


def make_path(df):
    line = LineString(
        df[["POSITION_X", "POSITION_Y"]]
        .astype(float)
        .itertuples(index=False, name=None)
    )
    return line


def make_hv_path(df):
    line = LineString(
        df[["POSITION_X", "POSITION_Y"]]
        .astype(float)
        .itertuples(index=False, name=None)
    )
    return hv.Path(line.coords)


def view_track(
    images: Iterable[np.ndarray], frame: int, track: pd.DataFrame, zoom=False
):
    layout = view_stacks(images, frame)

    path = make_hv_path(track)
    layout = (
        layout
        * path.opts(color="red", line_width=2)
        # * track.hvplot.scatter(x="POSITION_X", y="POSITION_Y").opts(
        #     color="red", marker="o"
        # )
    )
    spot_in_frame = track.query("frame == @frame")
    if not spot_in_frame.empty:
        perimeter = make_hv_perimeter(spot_in_frame)
        layout = layout * perimeter.opts(line_color="red", line_width=2, color=None)

        if zoom:
            cell_x, cell_y = spot_in_frame[["POSITION_X", "POSITION_Y"]].values[0]
            zoom_opts = hv.opts.Image(
                xlim=(cell_x - 30, cell_x + 30),
                ylim=(cell_y - 30, cell_y + 30),
                aspect=1,
            )
            layout = layout.opts(zoom_opts)

    return layout.cols(1)


def view_side_by_side(images: Iterable[np.ndarray], frame: int, track: pd.DataFrame):
    # TODO does not work, puts all images in one line
    return hv.Layout(
        [view_track(images, frame, track), view_track(images, frame, track, zoom=True)]
    ).opts(shared_axes=False)


def measure_spot(df, segmentation_map, stack):
    masked_array = np.ma.masked_where(
        segmentation_map[df.frame, ...] != df.image_id, stack[df.frame, ...]
    )
    return (
        masked_array.mean(),
        masked_array.std(),
    )


def measure_track(track: pd.DataFrame, segmentation_map, stack):
    # track[["mean_red", "std_red", "mean_green", "std_green"]] = track.apply(
    #     measure_spot,
    #     segmentation_map=segmentation_map,
    #     stack=stack,
    #     axis="columns",
    #     result_type="expand",
    # )
    result = track.apply(
        measure_spot,
        segmentation_map=segmentation_map,
        stack=stack,
        axis="columns",
        result_type="expand",
    )
    return result if not result.empty else pd.DataFrame(columns=["mean_red", "std_red"])


scaler = MinMaxScaler()


def view_red_green_track(
    images: Iterable[np.ndarray], red_track, green_track, frame=None
) -> hv.Layout:
    frame = frame if frame else max(red_track.frame.min(), green_track.frame.min())
    layout = view_track(images, frame, red_track)

    green_path = make_hv_path(green_track).opts(color="green")
    layout = layout * green_path
    green_spot = green_track.query("frame == @frame")
    if not green_spot.empty:
        green_perimeter = make_hv_perimeter(green_spot)
        layout = layout * green_perimeter.opts(
            line_color="green", line_width=2, color=None
        )

    return layout


def draw_fucci_measurement(
    df: pd.DataFrame,
    segmentation_map: np.ndarray,
    red_stack: np.ndarray,
    green_stack: np.ndarray,
    frame: int = None,
):
    df[["mean_red", "std_red"]] = measure_track(df, segmentation_map, red_stack)
    df[["mean_green", "std_green"]] = measure_track(df, segmentation_map, green_stack)
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
        df.hvplot(x="frame", y="mean_red", responsive=True, min_height=200).opts(
            color="red"
        )
        * df.hvplot.area(
            x="frame", y="low_red", y2="high_red", responsive=True, min_height=200
        ).opts(alpha=0.3, color="red")
        * df.hvplot(x="frame", y="mean_green", responsive=True, min_height=200).opts(
            color="green"
        )
        * df.hvplot.area(
            x="frame", y="low_green", y2="high_green", responsive=True, min_height=200
        ).opts(alpha=0.3, color="green")
        * hv.VLine(frame if frame else 0).opts(
            hv.opts.VLine(color="grey", line_width=3)
        )
    ).opts(responsive=True)


def draw_fucci_measurement_merged_track(
    df: pd.DataFrame,
    red_segmentation_map: np.ndarray,
    green_segmentation_map: np.ndarray,
    red_stack: np.ndarray,
    green_stack: np.ndarray,
    frame: int = None,
):
    df_red_segmap = df.query('source_track == "red"').copy()
    df_green_segmap = df.query('source_track == "green"').copy()

    df_red_segmap[["mean_red", "std_red"]] = measure_track(
        df_red_segmap, red_segmentation_map, red_stack
    )
    df_red_segmap[["mean_green", "std_green"]] = measure_track(
        df_red_segmap, red_segmentation_map, green_stack
    )
    df_green_segmap[["mean_red", "std_red"]] = measure_track(
        df_green_segmap, green_segmentation_map, red_stack
    )
    df_green_segmap[["mean_green", "std_green"]] = measure_track(
        df_green_segmap, green_segmentation_map, green_stack
    )

    try:
        df = pd.concat([df_red_segmap, df_green_segmap])
    except ValueError:
        df = pd.DataFrame(
            columns=["mean_red", "std_red", "mean_green", "std_green", *df.columns]
        )
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
        df.hvplot(x="frame", y="mean_red", responsive=True, min_height=200).opts(
            color="red"
        )
        * df.hvplot.area(
            x="frame", y="low_red", y2="high_red", responsive=True, min_height=200
        ).opts(alpha=0.3, color="red")
        * df.hvplot(x="frame", y="mean_green", responsive=True, min_height=200).opts(
            color="green"
        )
        * df.hvplot.area(
            x="frame", y="low_green", y2="high_green", responsive=True, min_height=200
        ).opts(alpha=0.3, color="green")
        * hv.VLine(frame if frame else 0).opts(
            hv.opts.VLine(color="grey", line_width=3)
        )
    ).opts(responsive=True)


class CartesianSimilarity:
    def __init__(self, tm_red: TrackmateXML, tm_green: TrackmateXML):
        self.tm_red = tm_red
        self.tm_green = tm_green

    @cache
    def calculate_metric(self, green_track_id, red_track_id):
        red_track_df = self.tm_red.trace_track(red_track_id)
        green_track_df = self.tm_green.trace_track(green_track_id)
        min_frame = max(red_track_df.frame.min(), green_track_df.frame.min())
        max_frame = min(red_track_df.frame.max(), green_track_df.frame.max())

        red_track_df = red_track_df.query("@min_frame <= frame <= @max_frame")
        green_track_df = green_track_df.query("@min_frame <= frame <= @max_frame")

        if len(red_track_df) < 5 or len(green_track_df) < 5:
            return np.inf

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

        return sse / (max_frame - min_frame)

    def calculate_metric_for_all_tracks(self):
        red_track_ids = self.tm_green.tracks.TrackID.unique().tolist()
        green_track_ids = self.tm_green.tracks.TrackID.unique().tolist()
        combinations = list(
            itertools.product(
                red_track_ids,
                green_track_ids,
            )
        )

        metrics = [
            self.calculate_metric(g, r)
            for r, g in tqdm.tqdm(combinations, desc="Calculating similarity metric")
        ]
        df = pd.DataFrame(columns=["red_track", "green_track"], data=combinations)
        df["metric"] = metrics

        return df.sort_values("metric").reset_index(drop=True)

    @cache
    def merge_tracks(self, red_track_id, green_track_id):
        red_track_df = self.tm_red.trace_track(red_track_id)
        green_track_df = self.tm_green.trace_track(green_track_id)

        overlap_frame_min = max(red_track_df.frame.min(), green_track_df.frame.min())
        overlap_frame_max = min(red_track_df.frame.max(), green_track_df.frame.max())

        overlap_red_frames = red_track_df.query(
            "@overlap_frame_min <= frame <= @overlap_frame_max+1"
        )
        overlap_green_frames = green_track_df.query(
            "@overlap_frame_min <= frame <= @overlap_frame_max+1"
        )
        # df = red_track_df.merge(
        #     green_track_df,
        #     on="frame",
        #     how="outer",
        #     suffixes=("_red", "_green"),
        #     indicator=True,
        # )

        rows = []
        for frame in range(overlap_frame_min, overlap_frame_max + 1):
            r = red_track_df.query("frame == @frame")
            g = green_track_df.query("frame == @frame")

            row = (r if g.empty else g).copy()
            row["source_track"] = "red" if g.empty else "green"
            if not r.empty and not g.empty:
                row = (r if r.AREA.values > g.AREA.values else g).copy()
                row["source_track"] = (
                    "red" if r.AREA.values > g.AREA.values else "green"
                )
                row["POSITION_Y"] = np.mean([r.POSITION_X, g.POSITION_X])
                row["POSITION_Y"] = np.mean([r.POSITION_Y, g.POSITION_Y])
            rows.append(row)
        yellow_frames = (
            pd.concat(rows)
            if rows
            else pd.DataFrame(columns=["source_track", *red_track_df.columns])
        )

        # yellow_frames = (
        #     pd.concat([overlap_red_frames, overlap_green_frames])
        #     .groupby("frame")[["frame", "POSITION_X", "POSITION_Y"]]
        #     .mean()
        #     .astype({"frame": "int"})
        # )

        red_frames = red_track_df.query(
            "frame < @overlap_frame_min or frame > @overlap_frame_max"
        ).copy()
        green_frames = green_track_df.query(
            "frame < @overlap_frame_min or frame > @overlap_frame_max"
        ).copy()

        yellow_frames["color"] = "yellow"
        red_frames["color"] = "red"
        red_frames["source_track"] = "red"
        green_frames["color"] = "green"
        green_frames["source_track"] = "green"

        return (
            pd.concat([red_frames, green_frames, yellow_frames])
            .reset_index(drop=True)
            .sort_values("frame")
        )


class CartesianSimilarityFromFile(CartesianSimilarity):
    def __init__(
        self, tm_red: TrackmateXML, tm_green: TrackmateXML, metric: pd.DataFrame
    ):
        super().__init__(tm_red, tm_green)
        self.metric_df = metric.sort_values("metric").reset_index(drop=True)

    @cache
    def calculate_metric(self, green_track_id, red_track_id):
        return self.metric_df.query(
            "green_track == @green_track_id and red_track == @red_track_id"
        ).metric.item()


class ViewType(Enum):
    individual = auto()
    merged = auto()


def view_merged_track(param, red_track, green_track, frame):
    # TODO view red/yellow/green track
    return view_red_green_track(
        param,
        red_track,
        green_track,
        frame=frame,
    ).opts(hv.opts.Polygons(line_color="yellow"))


class TrackViewer(param.Parameterized):
    current_red_track = param.Integer()
    current_green_track = param.Integer()
    frame = param.Integer()
    view_type = param.ObjectSelector(
        default=ViewType.individual.name, objects=[t.name for t in ViewType]
    )

    def __init__(
        self,
        red_stack,
        green_stack,
        tm_red: TrackmateXML,
        tm_green: TrackmateXML,
        red_segmentation_map,
        green_segmentation_map,
        metric: CartesianSimilarity = None,
        **params,
    ):
        super().__init__(**params)

        self.red_stack = red_stack
        self.green_stack = green_stack
        self.tm_red = tm_red
        self.tm_green = tm_green

        self.red_segmentation_map = red_segmentation_map
        self.green_segmentation_map = green_segmentation_map

        self.metric = metric if metric else CartesianSimilarity(tm_red, tm_green)
        self.df = self.metric.calculate_metric_for_all_tracks()

        self.view_type_wdgt = pn.widgets.RadioButtonGroup.from_param(
            self.param.view_type,
            opts={
                "name": "View Type",
                "button_type": "primary",  # TODO why is the button not taking the primary color?
            },
        )
        self.frame_wdgt = pn.widgets.IntSlider.from_param(self.param.frame)
        self.metric_wdgt = pn.widgets.Tabulator(self.df, page_size=7, show_index=False)
        self.metric_wdgt.on_click(self.metric_selected)

        top_red_track = self.df.loc[0].red_track.item()
        top_green_track = self.df.loc[0].green_track.item()
        track = self.metric.merge_tracks(
            red_track_id=top_red_track,
            green_track_id=top_green_track,
        )
        self.current_red_track = int(top_red_track)
        self.current_green_track = int(top_green_track)
        self.frame_wdgt.start, self.frame_wdgt.end = (
            track.frame.min().item(),
            track.frame.max().item(),
        )
        self.frame_wdgt.value = track.query('color == "yellow"').frame.min().item()

        # self.images = pn.Row(
        #     self.make_images(
        #         # self.df.iloc[0].red_track.item(),
        #         # self.df.iloc[0].green_track.item(),
        #     )
        # )
        # self.graphs = self.make_graphs()

    @pn.depends(
        "current_red_track",
        "current_green_track",
        "frame",
        "view_type",
        # watch=True,
    )
    def make_measurement(self):
        large = hv.Text(0, 0, "empty")

        if ViewType[self.view_type] is ViewType.merged:
            merged_track = self.metric.merge_tracks(
                self.current_red_track, self.current_green_track
            )
            large = draw_fucci_measurement_merged_track(
                merged_track,
                red_segmentation_map=self.red_segmentation_map,
                green_segmentation_map=self.green_segmentation_map,
                red_stack=self.red_stack,
                green_stack=self.green_stack,
                frame=self.frame,
            )

        if ViewType[self.view_type] is ViewType.individual:
            red_track = self.tm_red.trace_track(self.current_red_track)
            green_track = self.tm_green.trace_track(self.current_green_track)
            large = (
                draw_fucci_measurement(
                    red_track,
                    segmentation_map=self.red_segmentation_map,
                    red_stack=self.red_stack,
                    green_stack=self.green_stack,
                    frame=self.frame,
                )
                + draw_fucci_measurement(
                    green_track,
                    segmentation_map=self.green_segmentation_map,
                    red_stack=self.red_stack,
                    green_stack=self.green_stack,
                    frame=self.frame,
                )
            ).cols(1)

        # a = red_track.query("frame == @self.frame")
        # spot_in_frame = a if not a.empty else green_track.query("frame == @self.frame")
        # # if spot_in_frame.empty:
        # #     return pn.pane.HoloViews(large)
        #
        # cell_x, cell_y = spot_in_frame[["POSITION_X", "POSITION_Y"]].values[0]
        # zoom_opts = hv.opts.Image(
        #     xlim=(cell_x - 30, cell_x + 30),
        #     ylim=(cell_y - 30, cell_y + 30),
        #     aspect=1,
        # )
        # zoom = large.opts(zoom_opts, clone=True)
        # return pn.Row(pn.pane.HoloViews(large), pn.pane.HoloViews(zoom))
        return pn.pane.HoloViews(large)

    @pn.depends(
        "current_red_track",
        "current_green_track",
        "frame",
        "view_type",
        # watch=True,
    )
    def make_images(self):
        print(
            f"{self.current_red_track}, {self.current_green_track}, {self.frame}, {self.view_type}"
        )
        # red_track = self.tm_red.trace_track(red_track_id)
        # green_track = self.tm_green.trace_track(green_track_id)

        large = hv.Text(0, 0, "empty")

        if ViewType[self.view_type] is ViewType.merged:
            merged_track = self.metric.merge_tracks(
                self.current_red_track, self.current_green_track
            )

            large = view_merged_track(
                [self.red_stack, self.green_stack],
                merged_track,
                merged_track,
                frame=self.frame_wdgt.value,
            )

        if ViewType[self.view_type] is ViewType.individual:
            red_track = self.tm_red.trace_track(self.current_red_track)
            green_track = self.tm_green.trace_track(self.current_green_track)
            large = view_red_green_track(
                [self.red_stack, self.green_stack],
                red_track,
                green_track,
                frame=self.frame_wdgt.value,
            )

        # a = red_track.query("frame == @self.frame")
        # spot_in_frame = a if not a.empty else green_track.query("frame == @self.frame")
        # # if spot_in_frame.empty:
        # #     return pn.pane.HoloViews(large)
        #
        # cell_x, cell_y = spot_in_frame[["POSITION_X", "POSITION_Y"]].values[0]
        # zoom_opts = hv.opts.Image(
        #     xlim=(cell_x - 30, cell_x + 30),
        #     ylim=(cell_y - 30, cell_y + 30),
        #     aspect=1,
        # )
        # zoom = large.opts(zoom_opts, clone=True)
        # return pn.Row(pn.pane.HoloViews(large), pn.pane.HoloViews(zoom))
        return pn.pane.HoloViews(large)

    def metric_selected(self, event):
        print(
            f"Clicked cell in {event.column!r} column, row {event.row!r} with value {event.value!r}"
        )
        track = self.metric.merge_tracks(
            red_track_id=self.df.loc[event.row].red_track.item(),
            green_track_id=self.df.loc[event.row].green_track.item(),
        )
        self.frame_wdgt.start, self.frame_wdgt.end = (
            track.frame.min().item(),
            track.frame.max().item(),
        )
        self.current_red_track = int(self.df.loc[event.row].red_track.item())
        self.current_green_track = int(self.df.loc[event.row].green_track.item())
        self.frame_wdgt.value = track.query('color == "yellow"').frame.min().item()

        # self.images.objects[:] = [
        #     self.make_images(
        #         self.df.loc[event.row].red_track.item(),
        #         self.df.loc[event.row].green_track.item(),
        #     )
        # ]

    @pn.depends(
        "current_red_track",
        "current_green_track",
    )
    def make_top_label(self):
        return f"red track: {self.current_red_track}, green track: {self.current_green_track}"

    def view(self):
        return pn.Row(
            # self.images,
            self.make_images,
            pn.Column(
                self.make_top_label,
                self.view_type_wdgt,
                self.frame_wdgt,
                self.metric_wdgt,
                self.make_measurement,
            ),
        )
