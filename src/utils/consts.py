"""
    Author: SPDKH
    Date: Spring 2024
"""
from pathlib import Path
from src.utils.config import parse_args

ARGS = parse_args().parse_args()
print(ARGS)

# the path corresponding to current file directory
THIS_DIR = Path(__file__).__str__()
# the path corresponding to source path
SRC_DIR = Path(THIS_DIR).parents[1]
PROJ_DIR = Path(THIS_DIR).parents[2]
CONFIG_DIR = SRC_DIR / 'config'

SAMPLE_DIR = PROJ_DIR / 'sampled_ml_imgs'

IMG_SIZE = (400, 400, 3)
