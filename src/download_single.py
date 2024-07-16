"""
    Main Script to Run the program,
    load and configure the Google Map dataset
    © All rights reserved.
    author: spdkh
    date: July 2024, JacobsSensorLab
"""
import numpy as np
import pandas as pd
from src.utils import consts
from src.data.googlemap import GoogleMap
from src.utils import geo_helper


def main():
    """Function:
        Generates a GoogleMap object with specified parameters.
    Parameters:
        - args (list): List of arguments for the GoogleMap object.
        - map_type (str): Type of map to be generated (default: 'satellite').
        - data_dir (str): Directory to store the generated map data (default: current directory).
        - overlap (int): Amount of overlap between adjacent map tiles (default: 0).
    Returns:
        - aerial_data (GoogleMap): GoogleMap object with specified parameters.
    Processing Logic:
        - Generate GoogleMap object.
        - Set map type to 'satellite' if not specified.
        - Set data directory to current directory if not specified.
        - Set overlap to 0 if not specified.
    """
    args = consts.ARGS

    # if coords argument was given as a file instead of the actual coordinates:
    try:
        args.coords = np.asanyarray(pd.read_csv(args.coords, dtype=float, sep=' '))[1]
    except ValueError:
        pass

    bbox_m = geo_helper.get_map_dim_m(
        args.fov, args.coords[-1],
        args.aspect_ratio[0]/args.aspect_ratio[1]
        )
    bbox = geo_helper.calc_bbox_m(args.coords[:2],
                                  bbox_m)
    args.coords = tuple(np.array(bbox).flatten()) + (args.coords[-1],)
    aerial_data = GoogleMap(
        args=args,
        map_type='satellite',
        data_dir=args.data_dir,
        overlap=args.overlap
        )
    aerial_data.data_info['x'] = '.'
    aerial_data.check_data()


if __name__ == '__main__':
    main()
