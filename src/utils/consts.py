"""
    Author: SPDKH
    Date: Spring 1400
"""
import datetime
import pytz

from pathlib import Path
from src.utils.config import parse_args, update_args_with_json


ARGS = parse_args().parse_args(args=[])
update_args_with_json(ARGS, 'src/utils/config.json')

# the path corresponding to current file directory
THIS_DIR = Path(__file__).__str__()
# the path corresponding to source path
SRC_DIR = Path(THIS_DIR).parents[1]
PROJ_DIR = Path(THIS_DIR).parents[2]
CONFIG_DIR = SRC_DIR / 'config'
OUT_DIR = PROJ_DIR / Path(ARGS.result_dir)

if ARGS.log_name == 'current_date_and_time':
    ARGS.log_name = datetime.datetime.now(pytz.timezone('US/Central')).strftime("%d-%m-%Y_time%H%M")
CHK_FOLDER = '_'.join([ARGS.dataset,
                       ARGS.dnn_type,
                       ARGS.log_name])

WEIGHTS_DIR = OUT_DIR / CHK_FOLDER

SAMPLE_DIR = WEIGHTS_DIR / 'sampled_img'

LOG_DIR = OUT_DIR / 'graph' / WEIGHTS_DIR

small = (615, 515)
one_k = (1024, 768)
two_k = (2048, 1536)

IMG_SIZE = (400, 400, 3)
OVERLAP = 90