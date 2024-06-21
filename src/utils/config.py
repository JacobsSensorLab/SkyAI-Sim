"""
    parsing and configuration
    © All rights reserved.
    author: spdkh
    date: May 10, 2022
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
    # Define the arguments in a dictionary
    arguments = {
        'coords': {
            'type': float, 'nargs': '+',
            'default': [35.22, -90.07, 35.06, -89.73, 400],
            'help': 'Top left coords (lat, lon), bottom right coords, altitude ground level in feet'
        },
        'fov': {
            'type': float,
            'default': 78.8,
            'help': 'Diagonal field of view of the camera.'
        },
        'aspect_ratio': {
            'type': float, 'nargs': '+',
            'default': [4, 3],
            'help': 'Aspect ratio dimensions.'
        },
        'overlap': {
            'type': float,
            'default': 0,
            'help': 'Overlap of the camera field of view.'
        },
        'utm': {
            'type': str,
            'default': 'EPSG:32616',
            'help': 'UTM including zone information in EPSG format.'
        },
        'data_dir': {
            'type': str,
            'default': 'dataset/Memphis/',
            'help': 'Directory name to save the generated images'
        },
        'img_size': {
            'type': int, 'nargs': '+',
            'default': [400, 400, 3],
            'help': 'The size of batch'
        },
        'batch_size': {
            'type': int, 'choices': range(1, 128),
            'default': 8,
            'help': 'The size of batch'
        },
        'seed': {
            'type': int,
            'default': 2024,
            'help': 'Random seed value'
        }
    }
    # Load defaults from JSON and overwrite initial defaults if present
    json_defaults = update_args_with_json('src/utils/config.json')
    for arg, value in json_defaults.items():
        if arg in arguments:
            arguments[arg]['default'] = value

    # Add arguments to the parser with their default values
    for arg, options in arguments.items():
        parser.add_argument(f'--{arg}', **options)

    return parser


def update_args_with_json(json_file):
    """
    Updates default arguments with values from a JSON file.
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
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file: {e}")
            return {}
    else:
        print(f"Warning! Default file '{json_file}' not found.")
        return {}
