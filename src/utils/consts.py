"""
    © All rights reserved.
    author: spdkh
    date: sprint 2024, JacobsSensorLab
"""
import math
from pathlib import Path
from turtle import update
from src.utils.config import parse_args


def update_args(new_args=None):
    global ARGS
    if new_args is None:
        try:
            ARGS = parse_args().parse_args()
        except:
            # To be able to run in google colab
            ARGS = parse_args().parse_args(args=[])
    else:
        ARGS = new_args

ARGS = update_args()
print(ARGS)

# the path corresponding to current file directory
THIS_DIR = Path(__file__).__str__()
# the path corresponding to source path
SRC_DIR = Path(THIS_DIR).parents[1]
PROJ_DIR = Path(THIS_DIR).parents[2]
CONFIG_DIR = SRC_DIR / 'config'

SAMPLE_DIR = PROJ_DIR / 'sampled_ml_imgs'

IMG_SIZE = (400, 400, 3)

# center of mercator projection tile in pixels
tile_center_p = {'x': 128, 'y': 128}

# number of pixels per degree of longitude at zoom 0
pixel_per_degree = 256 / 360

# The mercator projection stretches the earth's surface into a flat map.
# This projection results in a vertical (y-axis) scaling factor
pixel_per_radian = 256 / (2 * math.pi)
