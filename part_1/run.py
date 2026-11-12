from pathlib import Path

import part_1

ROOT = Path(__file__).parent

part_1.part_1_main(
    ROOT / "dataset" / "census.csv",
    ROOT / "data",
)

part_1.part_1_plot(
    ROOT / "data",
    ROOT / "plots"
)
