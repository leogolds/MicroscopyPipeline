#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import holoviews as hv
import pandas as pd
import numpy as np

import trackmate_utils
import utils
import panel as pn

hv.extension("bokeh")

# %load_ext autoreload
# %autoreload 2


# In[2]:


# base_data_path = Path(r"data/fucci_60_frames")
# base_data_path = Path(r"D:\Data\full_pipeline_tests\left")
base_data_path = Path(r"D:\Data\full_pipeline_tests\left_60_frames")

red_stack_path = base_data_path / "red.tif"
green_stack_path = base_data_path / "green.tif"

base_model_path = Path(r"models")
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
metric_df = pd.read_hdf(base_data_path / "metric.h5", key="metric")
metric = trackmate_utils.CartesianSimilarityFromFile(tm_red, tm_green, metric_df)

# In[3]:


all_spots_df = metric.get_all_spots()
all_spots_df.head()


merged = metric.get_merged_tracks()
merged.head()


def get_binned_unmerged_tracks(bin_size=250, shape=red_stack.shape[1:]):
    red_unmerged, green_unmerged = metric.get_unmerged_red_green_tracks()
    result_df = pd.concat([red_unmerged, green_unmerged], ignore_index=True)
    x_interval_range = pd.interval_range(start=0, end=shape[1], freq=bin_size)
    result_df["x_grid_interval"] = pd.cut(result_df.POSITION_X, x_interval_range)
    result_df["x_bin"] = result_df.x_grid_interval.cat.rename_categories(
        [int(i.mid) for i in x_interval_range]
    )

    y_interval_range = pd.interval_range(start=0, end=shape[0], freq=bin_size)
    result_df["y_grid_interval"] = pd.cut(result_df.POSITION_Y, y_interval_range)
    result_df["y_bin"] = result_df.y_grid_interval.cat.rename_categories(
        [int(i.mid) for i in y_interval_range]
    )
    #
    result_df["y_bin"] = result_df["y_bin"].astype(np.float32)
    result_df["x_bin"] = result_df["x_bin"].astype(np.float32)

    return result_df


def get_binned_merged_tracks(bin_size=250, shape=red_stack.shape[1:]):
    # merged = metric.get_merged_tracks()
    result_df = metric.get_merged_tracks()
    x_interval_range = pd.interval_range(start=0, end=shape[1], freq=bin_size)
    result_df["x_grid_interval"] = pd.cut(result_df.POSITION_X, x_interval_range)
    result_df["x_bin"] = result_df.x_grid_interval.cat.rename_categories(
        [int(i.mid) for i in x_interval_range]
    )

    y_interval_range = pd.interval_range(start=0, end=shape[0], freq=bin_size)
    result_df["y_grid_interval"] = pd.cut(result_df.POSITION_Y, y_interval_range)
    result_df["y_bin"] = result_df.y_grid_interval.cat.rename_categories(
        [int(i.mid) for i in y_interval_range]
    )
    #
    result_df["y_bin"] = result_df["y_bin"].astype(np.float32)
    result_df["x_bin"] = result_df["x_bin"].astype(np.float32)

    return result_df


def calculate_flow_field():
    merged_df = get_binned_merged_tracks()
    unmerged_df = get_binned_unmerged_tracks()

    # Calculate flow field per track
    print("Calculating flow field")
    merged_df[["d_frame", "d_x", "d_y", "magnitude", "angle"]] = merged_df.groupby(
        "merged_track_id", group_keys=False
    ).progress_apply(trackmate_utils.track_flow)
    unmerged_df[["d_frame", "d_x", "d_y", "magnitude", "angle"]] = unmerged_df.groupby(
        ["source_track", "track_id"], group_keys=False
    ).progress_apply(trackmate_utils.track_flow)

    c = (
        pd.concat([merged_df, unmerged_df], ignore_index=True)
        .groupby(["frame", "x_bin", "y_bin"])
        .agg({"angle": "mean", "magnitude": "mean"})
        .reset_index()
    )

    return c


def visualize_flow_field(flow_field_df, frame=30, min_magnitude=0):
    aspect = red_stack.shape[1] / red_stack.shape[2]
    figure_width = 300

    frame_df = flow_field_df.dropna().query("frame == @frame")

    df = frame_df.query("magnitude > @min_magnitude")

    v_line = hv.VLine(
        x=min_magnitude,
    ).opts(color="red")
    histograms = (
        (
            frame_df.magnitude.hvplot.hist(
                title="Velocity", xlabel="velocity (um/frame)", xlim=(0, 20)
            )
            * v_line
            + df.angle.hvplot.hist(
                title="Direction", xlabel="angle (rad)", xlim=(-np.pi, np.pi)
            )
        )
        .cols(1)
        .opts(shared_axes=False)
    )

    a = hv.Image(
        np.flipud(red_stack[frame, ...]),
        bounds=(0, 0, red_stack.shape[2], red_stack.shape[1]),
    ).opts(
        frame_width=red_stack.shape[2], frame_height=red_stack.shape[1], cmap="gray"
    ) * df.hvplot.vectorfield(
        x="x_bin",
        y="y_bin",
        mag="magnitude_um",
        angle="angle"
        # x="x_bin", y="y_bin", mag="magnitude", angle="angle"
    ).opts(
        # color="magnitude",
        color="magnitude_um",
        xlim=(0, red_stack.shape[2]),
        ylim=(0, red_stack.shape[1]),
        frame_width=red_stack.shape[2],
        frame_height=red_stack.shape[1],
        colorbar=True,
        cmap="bgy",
    ).redim.range(
        magnitude_um=(0, 20)
    )

    return pn.Row(
        a.opts(frame_width=figure_width, frame_height=int(figure_width * aspect)),
        histograms,
        height=int(figure_width * aspect),
    )


def update():
    frame.value = frame.value + 1 if frame.value < 59 else 1


flow_field_df = calculate_flow_field()
flow_field_df["magnitude_um"] = flow_field_df.magnitude * 0.67
t = (
    flow_field_df.groupby(["frame", "x_bin", "y_bin"])
    .rolling(3)
    .agg({"angle": "mean", "magnitude": "mean"})
    .reset_index()
)

frame = pn.widgets.IntSlider(name="frame", value=1, start=1, end=59)
min_magnitude = pn.widgets.IntSlider(name="magnitude", value=1, start=0, end=25)
# # play = pn.widgets.Toggle(name='Play', button_type='success', value=False)
#
interactive = pn.bind(
    visualize_flow_field,
    flow_field_df=flow_field_df,
    # flow_field_df=t,
    frame=frame,
    min_magnitude=min_magnitude,
)
# # play.link(cb, bidirectional=True, value='running')
#
pn.Column(min_magnitude, frame, interactive).show()
#
# # In[24]:
#
#
# cb.running = False
#
# # In[25]:
#
#
# # In[ ]:
#
#
# # [hv.save(visualize_flow_field(flow_field_df=flow_field_df, frame=f),f'vid/frame_{f:02d}', 'gif') for f in range(1,60)]
#
#
# # In[ ]:
print()
