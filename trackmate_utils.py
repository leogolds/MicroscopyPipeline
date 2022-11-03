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
from typing import Iterable

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

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
    return track.apply(
        measure_spot,
        segmentation_map=segmentation_map,
        stack=stack,
        axis="columns",
        result_type="expand",
    )


scaler = MinMaxScaler()


def view_red_green_track(
    images: Iterable[np.ndarray], red_track, green_track, frame=None
):
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
        df.hvplot(x="frame", y="mean_red").opts(color="red")
        * df.hvplot.area(x="frame", y="low_red", y2="high_red").opts(
            alpha=0.3, color="red"
        )
        * df.hvplot(x="frame", y="mean_green").opts(color="green")
        * df.hvplot.area(x="frame", y="low_green", y2="high_green").opts(
            alpha=0.3, color="green"
        )
    )


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

        metrics = [self.calculate_metric(g, r) for r, g in tqdm.tqdm(combinations)]
        df = pd.DataFrame(columns=["red_track", "green_track"], data=combinations)
        df["metric"] = metrics

        return df
