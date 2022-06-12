from subprocess import ABOVE_NORMAL_PRIORITY_CLASS
from ui import ContrastEnhancement, ProbabilityMaps, MergeChannels, SegmentStack
import param
import panel as pn
import os

pn.extension("terminal", sizing_mode="scale_width", notifications=True, throttled=True)


class AnlysisApp(param.Parameterized):
    # stacks = param.ObjectSelector()
    # frame = param.Integer()
    # clip_limit = param.Number(default=0.03, bounds=(0, 1), step=0.01)
    # active_tool = param.Integer(default=0)

    def __init__(self, **params):
        self.contrast = ContrastEnhancement()
        self.probability = ProbabilityMaps()
        self.merge = MergeChannels()
        self.segment = SegmentStack()

        sections = (
            ("Contrast", self.contrast.get_controls()),
            ("Probability Maps (ilastik)", self.probability.get_controls()),
            ("Merge Channels", self.merge.get_controls()),
            ("Segment (CellPose)", self.segment.get_controls()),
        )
        self.controls = pn.Accordion(*sections, toggle=True, active=[0])

        # self.template.main = [pn.Column(self.create_dmap, sizing_mode="stretch_width")]
        self.main_windows = (
            self.contrast.get_main_window(),
            self.probability.get_main_window(),
            self.merge.get_main_window(),
            self.segment.get_main_window(),
        )

        self.template = pn.template.VanillaTemplate(
            title="H5 Viewer",
            sidebar=self.controls,
            main=pn.Column(self.main_windows[self.controls.active[0]]),
            sidebar_width=400,
        )

        super().__init__(**params)

    @param.depends("controls.active", watch=True)
    def on_change_tool(self):
        self.template.main.objects[0][:] = [self.main_windows[self.controls.active[0]]]

    def get_controls(self):
        return self.controls

    def get_main_window(self):
        pass

    def view(self):
        return self.template


app = AnlysisApp()
app.view().show()
