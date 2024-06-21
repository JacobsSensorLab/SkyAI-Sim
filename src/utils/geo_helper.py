"""
    Geolocation helper functions
    © All rights reserved.
    author: spdkh
    date: Aug 2023, JacobsSensorLab
"""
from typing import Tuple, List
import math
import geopy.distance
import pyproj
import numpy as np
import tensorflow as tf
import requests

from src.utils import consts


def get_static_map_image(
    data_dir: str, coords: Tuple[float, float],
    map_type: str, zoom: int=15, size: Tuple[int, int]=(640, 640),
    api_key: str=None, retry: int=10
    ):
    """Get a static map image from Google Maps API given
    latitude and longitude coordinates, map type, zoom level, size, and API key if available.
    Parameters:
        - data_dir (str): Direction where to save the loaded map data including datatype
        - coords (tuple): Latitude and Longitude coordinate.
        - lon (float): Longitude coordinate.
        - map_type (str): Type of map to retrieve.
        - zoom (int): Zoom level of the map, default is 15.
        - size (tuple): Size of the map image in pixels, default is (640, 640).
        - api_key (str): API key for Google Maps API, if available.
        - retry (int): Number of times to retry dowloading if errors happened
    Processing Logic:
        - Imports API key from hidden_file.py if available.
        - Constructs base URL and parameters for API call.
        - Removes labels from the map image.
        - Constructs final URL for API call.
        - Makes API call using requests library.
        - Saves the collected data to the given directory if available
        - Otherwise, specify the error and the URL for the error
    """
    try:
        if api_key is None:
            from src.hidden_file import api_key
    except ModuleNotFoundError:
        print('Warning: hidden_file.py is not available.')
        return
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{coords[0]},{coords[1]}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[1]}",
        "maptype": map_type,
        "style": "feature:all|element:labels|visibility:off",  # Remove labels
        "key": api_key,
    }

    final_url = base_url + "?" \
        + "&".join([f"{key}={value}" for key, value in params.items()])
    response = requests.get(base_url, params=params)

    for i in range(retry):
        if response.status_code == 200:
            with open(data_dir, "wb") as f:
                f.write(response.content)
            return
    if response.status_code != 200:
        raise ValueError(
            "Trial", i,
            "\nFailed to retrieve the image. Status code:", response.status_code,
            "URL:", final_url[:-1]
            )

# center of mercator projection tile in pixels
_C = {'x': 128, 'y': 128}

# number of pixels per degree of longitude at zoom 0
_J = 256 / 360

# The ercator projection stretches the earth's surface into a flat map.
# This projection results in a vertical (y-axis) scaling factor

_L = 256 / (2 * math.pi)
def calculate_bounding_box(
    center: Tuple[float, float], zoom: int,
    map_size: Tuple[int, int]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculates the bounding box coordinates for a given center point, zoom level, and map size.
    Parameters:
        - center (tuple):
            A tuple containing the latitude and longitude coordinates of the center point.
        - zoom (int):
            The zoom level of the map.
        - map_size (tuple):
            A tuple containing the width and height of the map in pixels.
    Returns:
        - tuple: A tuple containing the latitude and longitude coordinates of
                the top left and bottom right corners of the bounding box.
    Processing Logic:
        - Calculates the pixel size based on the zoom level.
        - Calculates the x and y coordinates of the center point.
        - Uses the x and y coordinates to calculate
            the top left and bottom right corners of the bounding box.
        - Converts the coordinates from pixel coordinates to latitude and longitude coordinates.
            The coordinates belong to top left and bottom right.

    Converted from JS to python using chatGPT
    Reference:
        https://stackoverflow.com/questions/44784839/calculate-bounding-box-of-static-google-maps-image
    """

    def clamp(value: float, min_value: float, max_value: float) -> float:
        """
        Ensures that a given value is within the specified range
        [min_value, max_value]
        """
        return max(min(value, max_value), min_value)

    def pt_to_lat_lon(pt: dict[str, float]) -> Tuple[float, float]:
        """
        Converts a point from pixel coordinates to latitude and longitude coordinates.
        """
        lon = (pt['x'] - _C['x']) / _J
        lat = math.degrees(math.asin(math.tanh((pt['y'] - _C['y']) / -_L)))
        return lat, lon

    # the width and height of the map in pixels, adjusted by the pixel_size
    # converted to mercator projection pixel values based on zoom
    pixel_size = pow(2, -(zoom + 1))
    half_pw_x = map_size[0] * pixel_size
    half_pw_y = map_size[1] * pixel_size

    a = clamp(
        math.sin(math.radians(center[0])),
        -(1 - 1E-15), 1 - 1E-15)

    # adjusted center point pixel coordinates
    cp = {
        'x': _C['x'] + center[1] * _J,
        'y': _C['y'] + 0.5 * math.log((1 + a) / (1 - a)) * -_L
    }

    top_left = pt_to_lat_lon({'x': cp['x'] - half_pw_x, 'y': cp['y'] - half_pw_y})
    bottom_right = pt_to_lat_lon({'x': cp['x'] + half_pw_x, 'y': cp['y'] + half_pw_y})

    return top_left, bottom_right


def reverse_bounding_box(
    top_left: Tuple[float, float],
    bottom_right: Tuple[float, float]
    ) -> Tuple[int, List[int]]:
    """
    Reverse the bounding box coordinates from lat/lon to zoom level and image size.
    If you know the top left and bottom right lat/lon,
    what would be the best zoom level that can fit the biggest possible image size (640) in it.
    Parameters:
        - top_left (tuple):
            Latitude and longitude of the top left corner of the bounding box.
        - bottom_right (tuple):
            Latitude and longitude of the bottom right corner of the bounding box.
    Returns:
        - zoom (int): Zoom level of the bounding box.
        - img_size (list): Image size of the bounding box in the format [width, height].
    Processing Logic:
        - Convert lat/lon coordinates to pixel points.
        - Calculate the pixel width and height of the bounding box.
        - Determine the zoom level based on the pixel width and height.
        - Calculate the final image size based on the zoom level.
    """
    def latlonToPt(lat, lon):
        a = min(
            max(
                math.sin(math.radians(lat)),
                -(1 - 1E-15)
                ),
            1 - 1E-15
            )
        cp = {
            'x': _C['x'] + lon * _J,
            'y': _C['y'] + 0.5 * math.log((1 + a) / (1 - a)) * -_L
        }
        return cp

    # Convert lat/lon coordinates to points
    cp_top_left = latlonToPt(top_left[0], top_left[1])
    cp_bottom_right = latlonToPt(bottom_right[0], bottom_right[1])

    # Calculate half pixel width and height
    half_pw_x = (cp_bottom_right['x'] - cp_top_left['x']) / 2
    half_pw_y = (cp_bottom_right['y'] - cp_top_left['y']) / 2

    # Initialize image size
    # This is the maximum pixel size available on Google Map
    # We get the high resolution first, then resize to the desired dimensions if needed
    img_size = 640

    # Determine width of the image based on the pixel dimensions
    # The bigger width along x or y will be the img_size
    # The other width is adjusted based on aspect ratio
    img_w = img_size if half_pw_x > half_pw_y else int(img_size * half_pw_y / half_pw_x)

    # Determine zoom level
    zoom = int(-math.log2(half_pw_x / img_w) - 1)

    # Calculate final image width and height based on zoom level
    scaling_factor = 2 ** (zoom + 1)
    img_w = int(half_pw_x * scaling_factor)
    img_h = int(half_pw_y * scaling_factor)

    return zoom, [img_w, img_h]


def overlapped(coords_a: tuple, coords_b: tuple, overlap: int = 25) -> bool:
    """Checking if two images with known label
        (top left and bottom right lat, long) are overlapped
        more than the desired overlap amount specified
    Parameters:
        - coords_a (tuple): Top left (lat, long) of the first image
        - coords_b (tuple): Buttom right(lat, lon) of the second image
        - overlap (int): [0-100] overlap percentage
    Returns:
        - bool: wether two images are overlapped by the percentage (True) or not (False)

    """
    return find_overlap(coords_a, coords_b) >= overlap


def find_overlap(coords_a: tuple, coords_b: tuple) -> int:
    """Calculates the percentage of overlap between two rectangles.
    Parameters:
        - coords_a (tuple): Coordinates of the first rectangle in the format
                            (top left lat, top left lon, buttom right lat, bottom right lon).
        - coords_b (tuple): Coordinates of the second rectangle in the format
                            (top left lat, top left lon, buttom right lat, bottom right lon).
    Returns:
        - overlap_percentage (int): Percentage of overlap between the two rectangles.
    Processing Logic:
        - Calculates the overlap in the x and y directions.
        - Calculates the area of each rectangle.
        - Calculates the percentage of overlap by dividing the overlap area
            by the total area of both rectangles.
        - Returns the overlap percentage as an int.
    """

    top_left_lat_a, top_left_lon_a, bottom_right_lat_a, bottom_right_lon_a = coords_a
    top_left_lat_b, top_left_lon_b, bottom_right_lat_b, bottom_right_lon_b = coords_b

    y_overlap = max(0, min(top_left_lat_a, top_left_lat_b)\
                    - max(bottom_right_lat_a, bottom_right_lat_b))
    x_overlap = max(0, min(bottom_right_lon_a, bottom_right_lon_b)\
                    - max(top_left_lon_a, top_left_lon_b))

    rect1_area = abs(bottom_right_lat_a - top_left_lat_a)\
                    * abs(bottom_right_lon_a - top_left_lon_a)
    rect2_area = abs(bottom_right_lat_b - top_left_lat_b)\
                    * abs(bottom_right_lon_b - top_left_lon_b)

    overlap_percentage =  x_overlap * y_overlap \
        / (rect1_area + rect2_area -  x_overlap * y_overlap)

    return int(overlap_percentage * 100)


def geo_calcs(data):
    """
        Gives information about the geolocation including
        minimum and maximum lat, long, alt, also
        area, width, height, etc in meters.
        The calculations are actual measurements based on the images already downloaded.
        the coordinates are based on the top left of the first image
        (slightly higher and lefter than top left of the map)
        and bottom right of the last image
        (slightly lower and righter than bottom right of the map)
    """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    print('Minimum:\n', data_min
                , '\nMaximum:\n', data_max)

    coords_ul = (data_min['Lat'], data_min['Lon'])
    coords_ur = (data_max['Lat'], data_min['Lon'])
    coords_dl = (data_min['Lat'], data_max['Lon'])
    coords_dr = (data_max['Lat'], data_max['Lon'])

    land_width = geopy.distance.geodesic(coords_ul, coords_ur).km
    land_height = geopy.distance.geodesic(coords_ul, coords_dl).km
    img_diagonal = geopy.distance.geodesic(coords_ul, coords_dr).km

    # only applicable if the images form a recangle overall
    land_area = land_width * land_height

    # only applicable if the images forming a rectangle do not overlap
    img_area = land_area / len(data)

    print('Area Diagonal Distance:', img_diagonal, ' Km',
                    '\nWidth =', land_width, 'Km',
                    '\nHeight =', land_height, 'Km',
                    '\nLand area = ', land_area, 'Km^2',
                    '\nArea covered by each image =', img_area, 'Km^2')

    return data_min, data_max


def meters2geo(
    center: Tuple[float, float], img_size: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Function: Converts the center point and image size from meters
        to geographic coordinates and returns the top left and bottom right coordinates.
    Parameters:
        - center (tuple): A tuple containing the center point coordinates in meters (x, y).
        - img_size (tuple): A tuple containing the width and height of the image in meters.
    Returns:
        - tl (tuple): A tuple containing the top left coordinates in latitude and longitude.
        - br (tuple): A tuple containing the bottom right coordinates in latitude and longitude.
    Processing Logic:
        - Convert center point from meters to UTM coordinates.
        - Calculate the bottom right and top left coordinates in UTM.
        - Convert the UTM coordinates to geographic coordinates.
        - Return the top left and bottom right coordinates.
    """

    # Compute half the width and height of the image
    img_w_m, img_h_m = np.array(img_size) / 2

    # Convert center point from geographic to UTM coordinates
    cxm, cym = geo2utm(center[0], center[1])

    # Calculate top left and bottom right UTM coordinates
    brm = (cxm + img_w_m, cym - img_h_m)
    tlm = (cxm - img_w_m, cym + img_h_m)

    # Convert UTM coordinates back to geographic coordinates
    tl = utm2geo(tlm[0], tlm[1])
    br = utm2geo(brm[0], brm[1])

    return tl, br


@tf.function
def geodist_loss_params(y_max, y_min):
    """
        Loss funcion to apply haversine distance difference between
        unnormalized inputs.
    """
    def geodist_loss(y_pred, y_true):
        y_pred = norm_helper.norm_undo(y_pred, y_max, y_min)
        y_true = norm_helper.norm_undo(y_true, y_max, y_min)

        lat1, lon1 = tf.unstack(y_pred, axis=-1)
        lat2, lon2 = tf.unstack(y_true, axis=-1)

        return haversine_distance((lat1, lon1, lat2, lon2))
    return geodist_loss


@tf.function
def haversine_distance(coords):
    """
    Calculates the haversine distance between two coordinates in meters.
    Parameters:
        - coords (list): List of four coordinates in decimal degrees [lat1, lon1, lat2, lon2].
    Returns:
        - distance (float): Haversine distance between the two coordinates in meters.
    Processing Logic:
        - Convert coordinates to radians.
        - Calculate differences in latitude and longitude.
        - Calculate haversine formula.
        - Convert distance to meters.
    """
    coords = [x * np.pi / 180.0 for x in coords]
    dlat = coords[2] - coords[0]
    dlon = coords[3] - coords[1]

    a = tf.math.sin(dlat / 2) ** 2  +\
        tf.math.cos(coords[0]) * tf.math.cos(coords[2]) * tf.math.sin(dlon / 2) ** 2
    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1 - a))

    distance = geopy.distance.EARTH_RADIUS * c
    return distance * 1000


def geo2utm(lat: float, lon: float, epsg: int = consts.ARGS.utm) -> Tuple[float, float]:
    """
    Converts geographic coordinates to UTM coordinates.
    Parameters:
        - lat (float): Latitude in decimal degrees.
        - lon (float): Longitude in decimal degrees.
        - epsg (int): EPSG code for desired UTM zone.
            Defaults to EPSG code for current UTM zone.
    Returns:
        - x (float): UTM easting coordinate.
        - y (float): UTM northing coordinate.
    Processing Logic:
        - Uses pyproj library for coordinate transformation.
        - EPSG code is used to determine UTM zone.
        - Coordinates are returned in meters.
    """

    x, y = pyproj.Transformer.from_crs(
        "EPSG:4326", epsg, always_xy=True
        ).transform(lon, lat)
    return x, y


def utm2geo(x: float, y: float, epsg: int = consts.ARGS.utm) -> Tuple[float, float]:
    """
    Converts UTM coordinates to geographic coordinates.
    Parameters:
        - x (float): UTM x-coordinate.
        - y (float): UTM y-coordinate.
        - epsg (int): EPSG code for UTM zone. Defaults to consts.ARGS.utm.
    Returns:
        - Tuple[float, float]: Geographic coordinates (latitude, longitude).
    Processing Logic:
        - Uses pyproj library for coordinate transformation.
        - EPSG code is required for accurate conversion.
        - Coordinates are returned in (latitude, longitude) format.
    """
    lon, lat = pyproj.Transformer.from_crs(
        epsg, "EPSG:4326", always_xy=True
        ).transform(x, y)
    return lat, lon
