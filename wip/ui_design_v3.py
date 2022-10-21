from datetime import datetime
from time import sleep
import panel as pn
from pathlib import Path
from typing import Callable
from wip.ui import ProbabilityMaps

pn.extension(
    sizing_mode="stretch_width",
    notifications=True,
)

path = Path(r"3pos/pos35/")
assert path.exists()


if __name__ == "__main__":
    viewer = ProbabilityMaps()
    row = pn.Row(
        viewer.get_controls,
        viewer.get_main_window,
    )
    row.show()
