import panel as pn
from pathlib import Path
from wip.ui import MergeChannels

pn.extension(
    sizing_mode="stretch_width",
    notifications=True,
)

path = Path(r"3pos/pos35/")
assert path.exists()


if __name__ == "__main__":
    viewer = MergeChannels()
    row = pn.Row(
        viewer.get_controls,
        viewer.get_main_window,
    )
    row.show()
