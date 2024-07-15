"""
    Path and Data Management helper functions
    © All rights reserved.
    author: spdkh
    date: 2023, JacobsSensorLab
"""
import os
import glob
import time

import geopy.point
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image


def import_module(module, class_name, *args, **kwargs):
    """
    Imports a specified module and class, and returns an instance of the class
    with the given arguments.
    Parameters:
        - module (str): Name of the module to be imported.
        - class_name (str): Name of the class to be instantiated.
        - *args (list): Optional arguments to be passed to the class constructor.
        - **kwargs (dict):
            Optional keyword arguments to be passed to the class constructor.
    Returns:
        - instance (object): An instance of the specified class.
    Processing Logic:
        - Creates a string representing the module and class names.
        - Imports the module using the string.
        - Uses the getattr() function to retrieve the specified class from the imported module.
        - Instantiates the class with the given arguments and returns the instance.
    """
    module_name = '.'.join(['src',
                            module,
                            class_name.lower()])
    import_module = __import__(module_name,
                                fromlist=[class_name])
    return getattr(import_module,
                        class_name)(*args, **kwargs)


# pylint: disable=W0212
def check_folder(log_dir):
    """
        check if directory does not exist,
        make it.

        params:

            log_dir: str
                directory to check
    """
    print('Checking folder:')
    if os.path.exists(log_dir):
        print('\t', log_dir, 'Folder Exists.')
        return True
    print('\tCreating Folder', log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return False


def find_files(path, ext):
    """
        params:

        path: str
            parent folder
        ext: str
            file extension

        returns: list
            list of directories of all files with
            given extention in the traverse directory
    """

    file_paths = []
    for folder_path in os.walk(path):
        file_paths.extend(glob.glob(folder_path[0] + '/*.' + ext))
    return file_paths


def metadata_read(img_path):
    """
        Read metadata embedded in JPG file
    :param img_path:
    :return:
    """
    img = Image.open(img_path)

    if 'exif' in img.info.keys():

        # build reverse dicts
        _tags_r = dict(((i, j) for j, i in TAGS.items()))
        _gpstags_r = dict(((i, j) for j, i in GPSTAGS.items()))

        # this merges gpsinfo as data rather than an offset pointer
        exifd = img._getexif()
        if "GPSInfo" in _tags_r.keys():
            gpsinfo = exifd[_tags_r["GPSInfo"]]

            lat = gpsinfo[_gpstags_r['GPSLatitude']],\
                  gpsinfo[_gpstags_r['GPSLatitudeRef']]
            long = gpsinfo[_gpstags_r['GPSLongitude']],\
                   gpsinfo[_gpstags_r['GPSLongitudeRef']]
            lat = str(lat[0][0]) + ' ' + str(lat[0][1]) + "m " \
                  + str(lat[0][1]) + 's ' + lat[1]
            long = str(long[0][0]) + ' ' + str(long[0][1]) + "m " \
                   + str(long[0][1]) + 's ' + long[1]

            meta_data = geopy.point.Point(lat + ' ' + long)

            return meta_data.format_decimal()

    pretty('Metadata not found!')
    return None


def pretty(*objects, sep=' ', end='\n', info=None, color="\033[93m"):
    if info == 'Warning!':
        color='\033[38;5;208m'
    print('\033[40m') # Set background to black
    reset = "\033[0m"  # Reset text color to default

    try:
        terminal_size = os.get_terminal_size().columns
    except:
        terminal_size = 75

    block_len = min(terminal_size,
                    max(
                        20 + max([len(i) for i in str(objects).split('\\n')]),
                        20 + len(str(info))
                        )
                    )
    header = 'Info:'
    print('-'*block_len)
    if info is not None:
         header += str(info)

    print(header.center(block_len))

    print('-'*block_len)

    print(color)

    print(*objects, sep=sep, end=end)
    print(reset + '\033[40m')

    print('-'*block_len)
    print(reset)


def pretty_args(args):
    text = ''
    index = 0
    for key, value in vars(args).items():
        if value is not None:
            index += 1
            text += "{:<20} = {:<10}".format(str(key), str(value))
            text += " " * 5 + '|' + " " * 5

            if index % 3 == 0:
                text += "\n"

    return text.replace(' ', '.')

def wait_for_files(expected_files, timeout=2):
    """
    Wait for expected files to appear in the directory.

    :param expected_files: List of filenames expected to appear.
    :param timeout: Maximum time to wait for the files (in seconds).
    """

    for _ in range(timeout * 10):
        if all(os.path.exists(directory) for directory in expected_files):
            return
        time.sleep(0.1)  # Wait for a second before checking again
    raise TimeoutError(f"Timeout: The expected files did not appear within {timeout} seconds.")