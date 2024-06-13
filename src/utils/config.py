"""
    author: Parisa Daj
    date: May 10, 2022
    parsing and configuration
"""
import os
import argparse
import json

import datetime
import pytz


def parse_args():
    """
        Define terminal input arguments
    Returns
    -------
    arguments
    """

    parser = argparse.ArgumentParser(
        prog='VBN Project',
        description='Multimodal Aerial Image translation / location estimation'
        )
    # general args
    parser.add_argument(
        "--coords", type=float, nargs='+',
        default=[35.22, -90.07, 35.06, -89.73, 400],
        help='Top left coords (lat, lon), buttom right coords, altitude ground level in feet'
        )

    parser.add_argument(
        '--fov', type=float, default=78.8,
        help='Diagonal field of view of the camera.'
        )

    parser.add_argument(
        "--aspect_ratio", type=float, nargs='+',
        default=[4, 3],
        help='top left latitude, longitude, buttom right latitude, longitude, altitude, hfov, vfov'
        )

    parser.add_argument(
        '--overlap', type=float, default=0,
        help='Diagonal field of view of the camera.'
        )

    parser.add_argument(
        '--utm', type=str, default='EPSG:32616',
        help='UTM including zone information in EPSG format.'
        )

    # Directories
    parser.add_argument(
        '--data_dir', type=str, default='dataset/Memphis/',
        help='Directory name to save the generated images'
        )

    log_name = datetime.datetime.now(
        pytz.timezone('US/Central')
        ).strftime("%d-%m-%Y_time%H%M")
    parser.add_argument(
        '--log_name', type=str, default=log_name,
        help='Desired name for the log file instead of date and time.'
        )

    parser.add_argument(
        '--batch_size', type=int, default=8,
        choices=range(1, 128),
        help='The size of batch'
        )

    parser.add_argument("--seed", type=int, default=1357)

    return parser


def update_args_with_json(args_, json_file):
    """
    Updates arguments with values from a JSON file.
    Parameters:
        - args_ (argparse.Namespace):
            Namespace object containing arguments.
        - json_file (str): Path to JSON file.
    Returns:
        - None: The function does not return anything.
    Processing Logic:
        - Load JSON file.
        - Iterate through key-value pairs.
        - Set attribute for each key-value pair.
    """
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            # Use json_data in further processing
            print(json_data)
        except IOError as e:
            print(f"Error opening or reading the file: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        return
    for key, value in json_data.items():
        setattr(args_, key, value)


if __name__=="__main__":
    args = parse_args()
