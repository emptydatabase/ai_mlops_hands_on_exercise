from pathlib import Path

import task1

ROOT = Path(__file__).parent

task1.part_1_plot(
    ROOT / "data",
    ROOT / "plots"
)
