from datetime import datetime
from time import sleep
import panel as pn
import param
import numpy as np
from numpy import uint8
from pathlib import Path
import h5py
import holoviews as hv
from holoviews.operation.datashader import regrid
from typing import Callable
from skimage.exposure import equalize_adapthist
from utils import create_dmap_from_image, docker_client, segment_frame
import pandas as pd

path = Path(r"3pos/pos35/")
assert path.exists()


class SegmentStack(param.Parameterized):
    stack_path = param.ObjectSelector()
    diameter_um = param.Integer(default=25, bounds=(10, 100))
    frame = param.Integer()
    # clip_limit = param.Number(default=0.03, bounds=(0, 1), step=0.01)

    process_stack_action = param.Action(lambda self: self.process_stack())
    interrupt_processing_action = param.Action(lambda self: self.interrupt_processing())

    def __init__(self, **params):
        super().__init__(**params)

        self.tqdm = pn.widgets.Tqdm()

        merged_stacks = list(path.glob("merged_probabilities.h5"))
        self.param.stack_path.objects = merged_stacks
        self.stack_path = self.param.stack_path.objects[0]

        self.stack = h5py.File(self.stack_path)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))

        frames, _, _, _ = self.stack.shape

        self.param.frame.bounds = (0, frames - 1)

        self.start_button = pn.Param(
            self.param,
            widgets={
                "process_stack_action": {
                    "type": pn.widgets.Button,
                    "name": "Process Stack",
                    "button_type": "success",
                },
            },
            parameters=["process_stack_action"],
            show_name=False,
        )
        self.stop_button = pn.Param(
            self.param,
            widgets={
                "interrupt_processing_action": {
                    "type": pn.widgets.Button,
                    "name": "Stop Processing",
                    "button_type": "danger",
                    "disabled": True,
                },
            },
            parameters=["interrupt_processing_action"],
            show_name=False,
        )

        self.processed_stack = None

    @param.depends("stack_path", watch=True)
    def load_file(self):
        # print("frame loaded")
        self.stack = h5py.File(self.stack_path)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))

        if len(self.stack.shape) == 5:
            frames, _, _, _, _ = self.stack.shape
            # self.stack = self.stack[:, 0, ..., 0]
        elif len(self.stack.shape) == 4:
            frames, _, _, _ = self.stack.shape
            # self.stack = self.stack[:, 0, ...]
        else:
            frames, _, _ = self.stack.shape
        self.frame = frames - 1 if frames - 1 < self.frame else self.frame
        self.param.frame.bounds = (0, frames - 1)

    @param.depends("frame", watch=True)
    def left_image(self):
        # print(f"left image shape: {self.stack.shape}")
        left_frame = self.stack[self.frame, 0, ...]

        # print("Refreshing image")
        img = hv.plotting.Image(left_frame).opts(
            # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    @param.depends("frame", watch=True)
    def right_image(self):
        # print(f"right image shape: {self.stack.shape}")
        right_frame = self.stack[self.frame, 0, ...]

        segmentation_mask, _ = segment_frame(right_frame, diameter=self.diameter_um)

        img = (
            hv.plotting.Image(segmentation_mask)
            .redim.range(z=(1, np.max(segmentation_mask)))
            .opts(
                # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
                colorbar=False,
                cmap="glasbey",
                clipping_colors={"min": "black"},
            )
        )
        return img

    @param.depends("frame", watch=True)
    def processed_image(self):
        if not self.processed_stack:
            try:
                frames, Z, Y, X, C = self.stack.shape
            except ValueError:
                frames, _, Y, X = self.stack.shape
            frame = np.zeros(shape=(Y, X), dtype=np.uint8)
        else:
            # frame = (self.processed_stack[self.frame, 0, ...] * 255).astype(np.uint8)
            frame = self.processed_stack[self.frame, 0, ...]

        # print("Refreshing image")
        img = hv.plotting.Image(frame).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    def process_stack(self):
        self.start_button.loading = True
        self.stop_button.disabled = False
        new_file_path = self.stack_path.parent / f"segmented.h5"
        new_table_path = self.stack_path.parent / f"segmented_table.h5"

        frames, _, Y, X = self.stack.shape

        with h5py.File(new_file_path, "w") as f:
            dataset = f.create_dataset(
                "data", shape=(frames, Y, X), dtype=np.uint16, chunks=True
            )

            for frame in self.tqdm(
                range(frames),
                desc="Segmentation",
                leave=True,
            ):
                # img = stack[frame, 0, :, :, 0]
                masks, center_of_mass = segment_frame(self.stack[frame, 0, ...])

                dataset[frame, :, :] = masks

                sub_df = pd.DataFrame(
                    center_of_mass,
                    columns=["center_y", "center_x"],
                )
                sub_df["timestep"] = frame

                # df = pd.concat([df, sub_df], ignore_index=True)
                sub_df.to_hdf(
                    new_table_path, key="table", format="table", append=True, mode="a"
                )

        self.processed_stack = h5py.File(new_file_path).get("data")
        self.stop_button.disabled = True
        self.start_button.loading = False

    def interrupt_processing(self):
        if not self.container:
            return

        self.container.stop()
        self.container = None

        self.stop_button.disabled = True
        self.start_button.loading = False

    def get_controls(self):
        return pn.Column(
            pn.Param(
                self.param,
                widgets={
                    "frame": pn.widgets.IntSlider,
                    "probability_threshold_left": pn.widgets.FloatSlider,
                    "probability_threshold_right": pn.widgets.FloatSlider,
                },
                parameters=[
                    "stack_path",
                    "frame",
                    "diameter_um",
                ],
            ),
            pn.Row(self.start_button, self.stop_button),
            self.tqdm,
        )

    def get_main_window(self):
        return pn.Row(
            pn.Row(
                create_dmap_from_image(self.left_image),
                create_dmap_from_image(self.right_image),
            ),
        )


class MergeChannels(param.Parameterized):
    left_stack_path = param.ObjectSelector()
    right_stack_path = param.ObjectSelector()
    probability_threshold_left = param.Number(default=0.9, bounds=(0, 1), step=0.01)
    probability_threshold_right = param.Number(default=0.9, bounds=(0, 1), step=0.01)
    frame = param.Integer()
    # clip_limit = param.Number(default=0.03, bounds=(0, 1), step=0.01)

    process_stack_action = param.Action(lambda self: self.process_stack())
    interrupt_processing_action = param.Action(lambda self: self.interrupt_processing())

    def __init__(self, **params):
        super().__init__(**params)

        self.tqdm = pn.widgets.Tqdm()

        probability_stacks = list(path.glob("*_Probabilities.h5"))
        self.param.left_stack_path.objects = probability_stacks
        self.left_stack_path = self.param.left_stack_path.objects[0]
        self.param.right_stack_path.objects = probability_stacks
        self.right_stack_path = self.param.right_stack_path.objects[1]

        self.left_stack = h5py.File(self.left_stack_path)
        self.left_stack = self.left_stack.get(
            "exported_data", self.left_stack.get("data")
        )
        self.right_stack = h5py.File(self.right_stack_path)
        self.right_stack = self.right_stack.get(
            "exported_data", self.right_stack.get("data")
        )
        frames, _, _, _, _ = self.left_stack.shape

        self.param.frame.bounds = (0, frames - 1)

        self.start_button = pn.Param(
            self.param,
            widgets={
                "process_stack_action": {
                    "type": pn.widgets.Button,
                    "name": "Process Stack",
                    "button_type": "success",
                },
            },
            parameters=["process_stack_action"],
            show_name=False,
        )
        self.stop_button = pn.Param(
            self.param,
            widgets={
                "interrupt_processing_action": {
                    "type": pn.widgets.Button,
                    "name": "Stop Processing",
                    "button_type": "danger",
                    "disabled": True,
                },
            },
            parameters=["interrupt_processing_action"],
            show_name=False,
        )

        self.processed_stack = None

    @param.depends("left_stack_path", watch=True)
    def load_left_file(self):
        # print("frame loaded")
        self.left_stack = h5py.File(self.left_stack_path)
        self.left_stack = self.left_stack.get(
            "exported_data", self.left_stack.get("data")
        )

        if len(self.left_stack.shape) == 5:
            frames, _, _, _, _ = self.left_stack.shape
            self.left_stack = self.left_stack[:, 0, ..., 0]
        else:
            frames, _, _ = self.left_stack.shape
        self.frame = frames - 1 if frames - 1 < self.frame else self.frame
        self.param.frame.bounds = (0, frames - 1)

    @param.depends("right_stack_path", watch=True)
    def load_right_file(self):
        # print("frame loaded")
        self.right_stack = h5py.File(self.right_stack_path)
        self.right_stack = self.right_stack.get(
            "exported_data", self.right_stack.get("data")
        )

        frames, _, _, _, _ = self.right_stack.shape

        self.frame = frames - 1 if frames - 1 < self.frame else self.frame
        self.param.frame.bounds = (0, frames - 1)

    @param.depends("frame", "probability_threshold_left", watch=True)
    def left_image(self):
        print(f"left image shape: {self.left_stack.shape}")
        left_frame = self.left_stack[self.frame, 0, ..., 0]
        masked_left_frame = np.where(
            left_frame > self.probability_threshold_left, left_frame, 0
        )
        # print("Refreshing image")
        img = hv.plotting.Image(masked_left_frame).opts(
            # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    @param.depends("frame", "probability_threshold_right", watch=True)
    def right_image(self):
        print(f"right image shape: {self.right_stack.shape}")
        right_frame = self.right_stack[self.frame, 0, ..., 0]
        masked_right_frame = np.where(
            right_frame > self.probability_threshold_right, right_frame, 0
        )
        img = hv.plotting.Image(masked_right_frame).opts(
            # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    @param.depends(
        "frame", "probability_threshold_left", "probability_threshold_right", watch=True
    )
    def merged_image(self):
        # print("Refreshing image")
        left_frame = self.left_stack[self.frame, 0, ..., 0]
        masked_left_frame = np.where(
            left_frame > self.probability_threshold_left, left_frame, 0
        )
        right_frame = self.right_stack[self.frame, 0, ..., 0]
        masked_right_frame = np.where(
            right_frame > self.probability_threshold_right, right_frame, 0
        )
        # merged_frame = np.sqrt(
        #     np.power(masked_left_frame, 2) + np.power(masked_right_frame, 2)
        # )
        # img = hv.plotting.Image(merged_frame).opts(
        #     colorbar=False,
        #     cmap="gray",
        # )
        try:
            img = hv.RGB(
                np.dstack(
                    [
                        masked_left_frame,
                        masked_right_frame,
                        np.zeros(shape=left_frame.shape),
                    ]
                )
            )
        except ValueError as e:
            print(e)
            img = hv.RGB(
                np.dstack(
                    [
                        np.ones(shape=left_frame.shape),
                        np.zeros(shape=left_frame.shape),
                        np.zeros(shape=left_frame.shape),
                    ]
                )
            )
        return img

    @param.depends("frame", watch=True)
    def processed_image(self):
        if not self.processed_stack:
            try:
                frames, Z, Y, X, C = self.left_stack.shape
            except ValueError:
                frames, Y, X = self.left_stack.shape
            frame = np.zeros(shape=(Y, X), dtype=np.uint8)
        else:
            # frame = (self.processed_stack[self.frame, 0, ...] * 255).astype(np.uint8)
            frame = self.processed_stack[self.frame, 0, ...]

        # print("Refreshing image")
        img = hv.plotting.Image(frame).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    def process_stack(self):
        self.start_button.loading = True
        self.stop_button.disabled = False
        new_file_path = self.left_stack_path.parent / f"merged_probabilities.h5"

        with h5py.File(new_file_path, "w") as f:
            data = f.create_dataset(
                "data",
                shape=self.left_stack.shape[:-1],
                dtype=self.left_stack.dtype,
                chunks=True,
            )

            for frame in self.tqdm(
                range(self.left_stack.shape[0]),
                desc="Merging Channels",
                leave=True,
            ):
                left_frame = self.left_stack[self.frame, 0, ..., 0]
                masked_left_frame = np.where(
                    left_frame > self.probability_threshold_left, left_frame, 0
                )
                right_frame = self.right_stack[self.frame, 0, ..., 0]
                masked_right_frame = np.where(
                    right_frame > self.probability_threshold_right, right_frame, 0
                )
                merged_frame = np.sqrt(
                    np.power(masked_left_frame, 2) + np.power(masked_right_frame, 2)
                )
                data[frame, ...] = merged_frame.astype(self.left_stack.dtype)

        self.processed_stack = h5py.File(path / f"merged_probabilites.h5").get("data")
        self.stop_button.disabled = True
        self.start_button.loading = False

    def interrupt_processing(self):
        if not self.container:
            return

        self.container.stop()
        self.container = None

        self.stop_button.disabled = True
        self.start_button.loading = False

    def get_controls(self):
        return pn.Column(
            pn.Param(
                self.param,
                widgets={
                    "frame": pn.widgets.IntSlider,
                    "probability_threshold_left": pn.widgets.FloatSlider,
                    "probability_threshold_right": pn.widgets.FloatSlider,
                },
                parameters=[
                    "left_stack_path",
                    "right_stack_path",
                    "frame",
                    "probability_threshold_left",
                    "probability_threshold_right",
                ],
            ),
            pn.Row(self.start_button, self.stop_button),
            self.tqdm,
        )

    def get_main_window(self):
        return pn.Row(
            pn.Column(
                create_dmap_from_image(self.left_image),
                create_dmap_from_image(self.right_image),
            ),
            pn.Column(
                create_dmap_from_image(self.merged_image),
                create_dmap_from_image(self.processed_image),
            ),
        )


class ProbabilityMaps(param.Parameterized):
    stack_path = param.ObjectSelector()
    model_path = param.ObjectSelector()
    frame = param.Integer()
    # clip_limit = param.Number(default=0.03, bounds=(0, 1), step=0.01)

    process_stack_action = param.Action(lambda self: self.process_stack())
    interrupt_processing_action = param.Action(lambda self: self.interrupt_processing())

    def __init__(self, **params):
        super().__init__(**params)

        self.tqdm = pn.widgets.Tqdm()

        self.param.stack_path.objects = list(path.glob("*_contrast_enhanced.h5"))
        self.stack_path = self.param.stack_path.objects[0]
        self.param.model_path.objects = list(path.glob("*.ilp"))
        self.model_path = self.param.model_path.objects[0]

        self.stack = h5py.File(self.stack_path)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))
        if len(self.stack.shape) == 5:
            frames, _, _, _, _ = self.stack.shape
            self.stack = self.stack[:, 0, ..., 0]
        else:
            frames, _, _ = self.stack.shape
        self.param.frame.bounds = (0, frames - 1)
        self.start_button = pn.Param(
            self.param,
            widgets={
                "process_stack_action": {
                    "type": pn.widgets.Button,
                    "name": "Process Stack",
                    "button_type": "success",
                },
            },
            parameters=["process_stack_action"],
            show_name=False,
        )
        self.stop_button = pn.Param(
            self.param,
            widgets={
                "interrupt_processing_action": {
                    "type": pn.widgets.Button,
                    "name": "Stop Processing",
                    "button_type": "danger",
                    "disabled": True,
                },
            },
            parameters=["interrupt_processing_action"],
            show_name=False,
        )

        self.terminal = pn.widgets.Terminal(
            options={"cursorBlink": True}, height=300, sizing_mode="stretch_width"
        )
        self.processed_stack = None

    @param.depends("stack_path", watch=True)
    def load_file(self):
        # print("frame loaded")
        self.stack = h5py.File(self.stack_path)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))

        if len(self.stack.shape) == 5:
            frames, _, _, _, _ = self.stack.shape
            self.stack = self.stack[:, 0, ..., 0]
        else:
            frames, _, _ = self.stack.shape
        self.frame = frames - 1 if frames - 1 < self.frame else self.frame
        self.param.frame.bounds = (0, frames - 1)

    @param.depends("frame", watch=True)
    def original_image(self):
        # print("Refreshing image")
        img = hv.plotting.Image(self.stack[self.frame, ...]).opts(
            # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    @param.depends("frame", watch=True)
    def processed_image(self):
        if not self.processed_stack:
            try:
                frames, Z, Y, X, C = self.stack.shape
            except ValueError:
                frames, Y, X = self.stack.shape
            frame = np.zeros(shape=(Y, X), dtype=np.uint8)
        else:
            frame = (self.processed_stack[self.frame, 0, ..., 0] * 255).astype(np.uint8)

        # print("Refreshing image")
        img = hv.plotting.Image(frame).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    def process_stack(self):
        self.start_button.loading = True
        self.stop_button.disabled = False

        command = [
            f"--project=/data/{self.model_path.name}",
            f"/data/{self.stack_path.name}",
        ]

        volumes = {
            # f"{model_path.parent.absolute()}": {"bind": "/model/", "mode": "ro"},
            f"{self.stack_path.parent.absolute()}": {"bind": "/data/", "mode": "rw"},
        }

        self.container = docker_client.containers.run(
            image="ilastik-container",
            detach=True,
            command=" ".join(command),
            volumes=volumes,
        )

        for line in self.container.logs(stream=True):
            self.terminal.write(line.decode("utf-8"))

        self.processed_stack = h5py.File(
            path / f"{self.stack_path.stem}_Probabilities.h5"
        ).get("exported_data")
        self.stop_button.disabled = True
        self.start_button.loading = False

    def interrupt_processing(self):
        if not self.container:
            return

        self.container.stop()
        self.container = None

        self.stop_button.disabled = True
        self.start_button.loading = False

    def get_controls(self):
        return pn.Column(
            pn.Param(
                self.param,
                widgets={"frame": pn.widgets.IntSlider},
                parameters=["stack_path", "model_path", "frame", "clip_limit"],
            ),
            pn.Row(self.start_button, self.stop_button),
            self.tqdm,
        )

    def get_main_window(self):
        return pn.Row(
            pn.Column(
                "# Original",
                create_dmap_from_image(self.original_image),
                "# Probability Map",
                create_dmap_from_image(self.processed_image),
            ),
            self.terminal,
        )


class ContrastEnhancement(param.Parameterized):
    stack_paths = param.ObjectSelector()
    frame = param.Integer()
    clip_limit = param.Number(default=0.03, bounds=(0, 1), step=0.01)

    process_stack_action = param.Action(lambda self: self.process_stack())
    processing = param.Boolean(precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)

        self.tqdm = pn.widgets.Tqdm()

        self.param.stack_paths.objects = list(path.glob("*.h5"))
        self.stack_paths = self.param.stack_paths.objects[0]

        self.stack = h5py.File(self.stack_paths)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))
        if len(self.stack.shape) == 5:
            frames, _, _, _, _ = self.stack.shape
            self.stack = self.stack[:, 0, ..., 0]
        else:
            frames, _, _ = self.stack.shape
        self.param.frame.bounds = (0, frames - 1)
        self.button = pn.Param(
            self.param,
            widgets={
                "process_stack_action": {
                    "type": pn.widgets.Button,
                    "name": "Process Stack",
                    "button_type": "success",
                }
            },
            parameters=["process_stack_action"],
            show_name=False,
        )

    @param.depends("stack_paths", watch=True)
    def load_file(self):
        # print("frame loaded")
        self.stack = h5py.File(self.stack_paths)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))

        if len(self.stack.shape) == 5:
            frames, _, _, _, _ = self.stack.shape
            # self.stack = self.stack[:, 0, ..., 0]
        else:
            frames, _, _ = self.stack.shape
        self.frame = frames - 1 if frames - 1 < self.frame else self.frame
        self.param.frame.bounds = (0, frames - 1)

    @param.depends("frame", watch=True)
    def original_image(self):
        print(f"original image {self.stack.shape}")
        img = hv.plotting.Image(self.stack[self.frame, ...]).opts(
            # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    @param.depends("frame", "clip_limit", watch=True)
    def contrast_enhanced_image(self):
        # print("Refreshing image")
        equalized = equalize_adapthist(
            self.stack[self.frame, ...], clip_limit=self.clip_limit
        )
        img = hv.plotting.Image(equalized * 255).opts(
            # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    def process_stack(self):
        # TODO ilastik expects TZYXC, output correct shape
        if self.processing:
            return

        self.button.loading = True
        new_file_path = (
            self.stack_paths.parent / f"{self.stack_paths.stem}_contrast_enhanced.h5"
        )

        frames, Y, X = self.stack.shape

        with h5py.File(new_file_path, "w") as f:
            data = f.create_dataset(
                "data",
                shape=(frames, 1, Y, X, 1),
                dtype="uint8",
                chunks=True,
            )
            for frame in self.tqdm(
                range(self.stack.shape[0]),
                desc="Enhancing Contrast",
                leave=True,
            ):
                data[frame, 0, ..., 0] = (
                    equalize_adapthist(
                        self.stack[frame, ...], clip_limit=self.clip_limit
                    )
                    * 255
                ).astype(uint8)
        self.button.loading = False

        pn.state.notifications.info(
            f"Contrast enhanced stack created at {new_file_path}"
        )

    def get_controls(self):
        return pn.Column(
            pn.Param(
                self.param,
                widgets={"frame": pn.widgets.IntSlider},
                parameters=["stack_paths", "frame", "clip_limit"],
            ),
            self.button,
            self.tqdm,
        )

    def get_main_window(self):
        return pn.Row(
            pn.Column(
                "# Original",
                create_dmap_from_image(self.original_image),
            ),
            pn.Column(
                "# Contrast Enhanced",
                create_dmap_from_image(self.contrast_enhanced_image),
            ),
        )
