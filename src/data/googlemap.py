"""
    Load VBN data generated in simulation
    © All rights reserved.
    author: spdkh
    date: June 2023, JacobsSensorLab
"""
import os
import glob
from pathlib import Path
import geopy
from matplotlib.streamplot import OutOfBounds
from tqdm import tqdm
import numpy as np
import pandas as pd
from natsort import natsorted
import skimage.measure
import tensorflow as tf

from src.utils import data_helper, geo_helper, preprocess
from src.utils.beauty import pretty
from src.data.vbn import VBN
from src.data.imagedata import ImageData


class GoogleMap(VBN, ImageData):
    """
        Load Googlemap API data
    """
    def __init__(self, **kwargs):
        """Function:
            def __init__(self, **kwargs):
                Initializes a Googlemap object.
                Parameters:
                    - kwargs (dict): A dictionary of keyword arguments.
                        - args: args from parsearg or config.json file
                        - data_dir: data directory string (withouth overlap and )
                        - map_type: choices between 'roadmap', 'satellite', 'terrain', etc.
                        - overlap: int ranges from 0 to 99
                Returns:
                    - None.
                Processing Logic:
                    - Sets the map type and overlap based on the keyword arguments.
                    - Sets the data directory based on the keyword arguments.
                    - Calculates the aspect ratio and altitude in meters.
                    - Calculates the diagonal in meters in the image using the fov and aspect ratio.
                    - Calculates the width and height in meters from the diagonal.
                    - Checks if the data directory exists and if the data is valid.
                    - Sets the input directory by finding all .jpg files in the data directory.
                    - Checks if the input directory exists.
        """
        ImageData.__init__(self, **kwargs)
        VBN.__init__(self, **kwargs)
        self.log = pd.DataFrame(
            index=['y/lat', 'x/lon', 'z/agl', 'total/area'],
            columns = ['TL',
                       'BR',
                       'Center',
                       'TL (UTM)',
                       'BR (UTM)',
                       'Map Size (m)',
                       'Map Size (pixel/zoom)',
                       '# Raster Images',
                       '1 Image Size (m)',
                       '1 Image Size (pixel/zoom)',
                       ])
        self.data_info = {'x': 'images'}

        self.input_dim = kwargs['args'].img_size

        self.map_type  = kwargs['map_type'] if 'map_type' in kwargs\
            else 'satellite'
        self.overlap = kwargs['overlap'] if 'overlap' in kwargs\
            else kwargs['args'].overlap

        # This is helpful if we want multiple Googlemap objects in the same program
        data_dir = kwargs['data_dir'] if 'data_dir' in kwargs\
            else kwargs['args'].data_dir

        # Add dataset information to the name of the dataset folder
        data_dir += self.map_type
        data_dir += '_' + str(self.overlap)
        data_dir += '_' + str(self.args.coords)[1:-1].replace(', ', '_')
        self.data_dir = Path(data_dir)

        self.log['1 Image Size (m)'].iloc[:-1] =\
            np.round(
                geo_helper.get_map_dim_m(
                self.args.fov,
                self.args.coords[4],
                self.args.aspect_ratio[0] / self.args.aspect_ratio[1]
                ),
                2)
        self.log['1 Image Size (m)'].iloc[-1] =\
            round(self.log['1 Image Size (m)'].iloc[0] *\
                self.log['1 Image Size (m)'].iloc[1], 2)
        self.log['TL'].iloc[:2] = self.args.coords[:2]
        self.log['BR'].iloc[:2] = self.args.coords[2:4]

        data_helper.check_folder(self.data_dir)

        self.check_data()

        self.input_dir = data_helper.find_files(self.data_dir /
                                                 self.data_info['x'],
                                                 'jpg')
        data_helper.check_folder(self.data_dir / self.data_info['x'])

    def check_data(self):
        """
            Generate GoogleMap data from a given big picture map
            These calculations are theoretical measurements,
            based on TL and BR coordinates of the map,
            slightly different than the area captured by images in geo_helper.geo_calcs

        :return:
        """
        top_left = self.args.coords[0], self.args.coords[1]
        bottom_right = self.args.coords[2], self.args.coords[3]

        map_zoom, map_size =\
            geo_helper.get_zoom_from_bounds(top_left, bottom_right)
        self.log['Map Size (pixel/zoom)'].iloc[:-1] = map_size + [map_zoom]

        x_width_m, y_width_m, _, _ = self.log['1 Image Size (m)']

        center_lat = (top_left[0] + bottom_right[0]) / 2
        center_lon = (top_left[1] + bottom_right[1]) / 2
        self.log['Center'].iloc[:2] = center_lat, center_lon
        ## Convert geolocation of the raster corners to utm
        # TL point
        self.log['TL (UTM)'].iloc[:2] =\
            np.round(geo_helper.geo2utm(top_left[0], top_left[1]), 2)

        # TL point
        self.log['BR (UTM)'].iloc[:2] =\
            np.round(geo_helper.geo2utm(bottom_right[0], bottom_right[1]), 2)

        ## Start Raster from TL of the map
        # get coordinates of the corners
        # of the top most left image in the raster mission
        tl, br = geo_helper.meters2geo(
            center=top_left,
            img_size=self.log['1 Image Size (m)'].iloc[:2])
        raster_zoom, im_size = geo_helper.get_zoom_from_bounds(tl, br)
        self.log['1 Image Size (pixel/zoom)'].iloc[:-1] = im_size + [raster_zoom]

        ## Convert geolocation of the raster corners to utm
        # TL point
        map_tlm = geo_helper.geo2utm(top_left[0], top_left[1])

        # TL point
        map_brm = geo_helper.geo2utm(bottom_right[0], bottom_right[1])
        self.log['Map Size (m)'].iloc[:2] = np.abs(np.subtract(map_tlm[:2], map_brm[:2])).round(2)
        self.log['Map Size (m)'].iloc[-1] =\
            round(self.log['Map Size (m)'].iloc[0] *\
                self.log['Map Size (m)'].iloc[1], 2)

        self.log['# Raster Images'].iloc[:2] = (
            ((self.log['Map Size (m)'].iloc[:2] - self.log['1 Image Size (m)'].iloc[:2])
            / (self.log['1 Image Size (m)'].iloc[:2] * (1 - self.overlap)))
            .astype(int) + 1
        )
        self.log['# Raster Images'].iloc[-1] =\
            self.log['# Raster Images'].iloc[0] *\
                self.log['# Raster Images'].iloc[1]


        pretty('[INFO] Data detailed values before download.\n',
               'TL = Top Left, BR = Bottom Right\n',
               self.log.to_string(), info=self)

        map_name = "map_" + '_'.join(
            list(map(str, self.args.coords))
        ) + ".jpg"
        if map_name not in os.listdir(self.data_dir):
            geo_helper.get_static_map_image(
                self.data_dir / map_name,
                [center_lat, center_lon],
                map_type=self.map_type,
                zoom=map_zoom,
                size=map_size
                )
        else:
            pretty('Map image is available in', self.data_dir, 'as', map_name,
                   info=self)

    def config(self, download_raster=True):
        """
        Downloads the data, configures, then do the geolocation calculations.
        Parameters:
            - self (object): The object being passed in.
        Returns:
            - None: The function does not return anything.
        Processing Logic:
            - Calculate the minimum and maximum of the original output.
            - Clean up the data.

        todo: more conditions for cleanup
        """
        if download_raster:
            self.complete_download()

        super().config()
        self.org_out_min, self.org_out_max =\
              geo_helper.geo_calcs(self.labels)
        self.cleanup_data()

    def complete_download(self):
        """
        Finds the latest file downloaded from the data directory
        """
        sorted_imgs = natsorted(
            glob.glob(
                os.path.join(self.data_dir / 'images', "*.jpg")
            ),
            reverse=True
        )
        latest_image_name = sorted_imgs[0].split('/')[-1] if len(sorted_imgs) else -1
        print('Latest Image Name:', latest_image_name)
        self.gen_raster_from_map((self.args.coords[0], self.args.coords[1]),
                                (self.args.coords[2], self.args.coords[3]),
                                overlap=self.overlap,
                                last_img_name=latest_image_name)

    def config_output(self):
        if os.path.exists(self.data_dir / 'meta_data.csv'):
            pretty('Found the metadata file...',
               info=self)
            self.meta_df = pd.read_csv(self.data_dir / 'meta_data.csv')
        else:
            pretty('Generating the metadata file...',
               info=self)
            meta_data = []
            meta_data = [[os.path.basename(path)] \
                + list(map(
                    float,
                    os.path.basename(path).split('.jpg')[0].split('_'))) for path in self.input_dir
                ]
            self.meta_df = pd.DataFrame(
                meta_data,
                columns=['img_names', 'columns', 'row', 'Lat', 'Lon', 'Alt']
            )
            #todo: calc entropy for the same map type
            #todo: apply cleaning based on roadmap data if available in data cleaning func
            road_dir = self.data_dir
            if self.map_type != 'roadmap':
                road_folder_name = self.data_dir.name.replace(
                    self.map_type, 'roadmap'
                )
                road_dir = self.data_dir.parents[0] / road_folder_name
            road_dir = road_dir / self.data_info['x']

            self.meta_df['entropies'] = self.meta_df.apply(
                self.calc_entropy(road_dir), axis=1
            )
            self.meta_df.to_csv(self.data_dir / 'meta_data.csv')

        pretty('All metadata:\n', self.meta_df,
               info=self)

        self.labels = self.meta_df.loc[:, ['Lat', 'Lon', 'Alt']]

    def calc_entropy(self, dir):
        def entropy_per_row(row):
            img = self.imread(Path(dir) / Path(row['img_names']))
            entropy = skimage.measure.shannon_entropy(img)
            return entropy
        return entropy_per_row

    def gen_raster_from_map(self,
                        top_left_coords: tuple,
                        bottom_right_coords: tuple,
                        overlap: int = 0,
                        last_img_name: str = -1,
                        margin: int = 20):

        """
        Generate a raster of map images based on specified coordinates and other parameters.
        Parameters:
            - top_left_coords (tuple): Geolocation coordinates (latitude, longitude)
                for the top-left corner of the raster.
            - bottom_right_coords (tuple): Geolocation coordinates (latitude, longitude)
                for the bottom-right corner of the raster.
            - overlap (int, optional): Percentage overlap between images in the raster.
                Default is 0.
            - last_img_name (str, optional):
                Name of the last downloaded image to resume the download process, if applicable.
                Default is -1.
            - margin (int, optional): Percentage of margin to add to the image for removing marks.
                Default is 20.
        Returns:
            - None: The function does not return any value but downloads the raster images to the specified directory.
        Example:
            - gen_raster_from_map((34.052235, -118.243683), (34.040713, -118.246769))
        """
        ## Convert geolocation of the raster corners to utm
        # TL point
        map_tlm = geo_helper.geo2utm(top_left_coords[0], top_left_coords[1])

        # TL point
        map_brm = geo_helper.geo2utm(bottom_right_coords[0], bottom_right_coords[1])

        ## Start Raster from TL of the map
        # get coordinates of the corners
        # of the top most left image in the raster mission
        tl, br = geo_helper.meters2geo(center=top_left_coords,
                                       img_size=self.log['1 Image Size (m)'][:2])
        raster_zoom, im_size = geo_helper.get_zoom_from_bounds(tl, br)

        # Convert coordinates to UTM
        tlm = geo_helper.geo2utm(tl[0], tl[1])
        brm = geo_helper.geo2utm(br[0], br[1])
        x_orig = tlm[0]

        # Width of each raster image along x, y in meters
        raster_wy = np.abs(tlm[1] - brm[1])
        raster_wx = np.abs(tlm[0] - brm[0])

        ## Add vertical margins
        # so that the google mark can be removed later after download
        # the same margin value has to be applied to the preprocess_image method
        im_size[1] += int(im_size[1] * margin/100)

        # Calculate number of images fit in the map in a raster mission
        map_size_m = np.abs(np.subtract(map_tlm[:2], map_brm[:2]))
        n_images_x = 1 + int((map_size_m[0] - raster_wx)\
                    /(((100 - overlap) / 100) * raster_wx))
        n_images_y = 1 + int((map_size_m[1] - raster_wy)\
                    /(((100 - overlap) / 100) * raster_wy))

        # Calculate what has already been downloaded and what is left
        if last_img_name != -1:
            # Extract coordinates and IDs from file name and convert to UTM
            last_x, last_y, lat_i, lon_j, _ = last_img_name[:-4].split('_')
            last_x, last_y = int(last_x), int(last_y)
            lat_i, lon_j = float(lat_i), float(lon_j)
            x, y = geo_helper.geo2utm(lat_i, lon_j)

            # last x and y start from 0
            if [last_x + 1, last_y + 1] == [n_images_x, n_images_y]:
                pretty('All data already downloaded.', info=self)
                return

            pretty('Downloading the rest of Google Map images from \nx = ',
                   last_x + 1, ' / ', n_images_x,
                   '\ny = ', last_y + 1, ' / ', n_images_y, info=self)
            # Determine the UTM coordinates and ID of the nex image to download
            if last_x < n_images_x - 1:
                last_x += 1
                x += raster_wx * (100 - overlap) / 100
            else:
                last_x = 0
                last_y += 1
                y -= raster_wy * (100 - overlap) / 100
                x = x_orig
            lat_i, lon_j = geo_helper.utm2geo(x, y)
        else:
            pretty('Downloading All Google Map images...', info=self)
            last_x, last_y = 0, 0
            lat_i, lon_j = top_left_coords
            x, y = tlm

        i, j = last_x, last_y

        if n_images_x * n_images_y > 5000:
            print('Warning!\nNumber of images exceed the publishable limit. Read more at:')
            print('https://about.google/brand-resource-center/products-and-services/geo-guidelines/#required-attribution/')
        data_helper.check_folder(self.data_dir / 'images')

        response = input("Do you want to proceed? (y/yes): ").strip().lower()
        if response in ["y", "yes"]:
            print("Confirmed.")
            with tqdm(position=0, leave=True, total=n_images_x*n_images_y) as pbar:
                pbar.update((n_images_x*last_y) + last_x)
                for j in range(last_y, n_images_y):
                    i = last_x if j == last_y else 0
                    while i < n_images_x:
                        if lat_i < bottom_right_coords[0] - 0.02 or lon_j > bottom_right_coords[1] + 0.02:
                            raise OutOfBounds(
                                "Exceeding BR Coordinate limits; BR coordinates:",
                                bottom_right_coords,
                                'Current coordinates:', lat_i, lon_j,
                                'Indices:', i, j
                            )

                        out_name = str(i) + '_' \
                                + str(j) + '_' \
                                + str(lat_i) \
                                + '_' + str(lon_j) \
                                + '_' + str(raster_zoom) + '.jpg'

                        output_dir = self.data_dir / 'images' / out_name
                        geo_helper.get_static_map_image(
                            output_dir,
                            [lat_i, lon_j],
                            map_type=self.map_type,
                            zoom=raster_zoom,
                            size=im_size
                            )
                        x += raster_wx * (100 - overlap) / 100
                        _, lon_j = geo_helper.utm2geo(x, y)

                        pbar.update()
                        i += 1
                    y -= raster_wy * (100 - overlap) / 100
                    x = x_orig
                    lat_i, lon_j = geo_helper.utm2geo(x, y)
        else:
            pretty("Not Proceeding the download. Continuing without the download...",
                   info=self)

        print('\t Number of rows and columns:', i, j)


    def cleanup_data(self, entropy_thr=2.1):
        """
        filter out unused data using entropy if map type is roadmap.
        """
        thr_q = self.meta_df['entropies'] >= entropy_thr

        self.labels = self.meta_df[thr_q][self.labels.columns]
        self.input_dir = list(self.meta_df[thr_q]['img_names'].apply(
            self.add_parent_dir())
                              )
        print(self.labels.sample(5), len(self.input_dir))


    def preprocess_image(self, image, margin=20):
        """
        Preprocess an input image by cropping and resizing.
        Parameters:
            - image (tf.Tensor): Input image to be preprocessed as a TensorFlow tensor.
            - margin (int, optional): Margin size used for cropping in percentage.
                This should preferrably have the same value as in gen_raster_from_map method.
                Default is 20
        Returns:
            - tf.Tensor: Preprocessed image tensor after cropping and resizing.
        Example:
            - preprocess_image(image_tensor, 20) -> preprocessed_image_tensor
        """

        # Convert the float values to integer by multiplying with the image dimensions
        image_shape = tf.cast(tf.shape(image), tf.float32)

        # crop the google sign from the bottom
        image = tf.image.crop_to_bounding_box(
            image,
            tf.cast(image_shape[0] * 0, dtype=tf.int32),
            tf.cast(image_shape[1] * margin/200, dtype=tf.int32),
            tf.cast(image_shape[0] * 1, dtype=tf.int32),
            tf.cast(image_shape[1] * 1-margin/200, dtype=tf.int32)
        )

        # resize to the desired shape
        image = tf.image.resize(image, self.input_dim[0:2])

        # equalize histogram if roadmap
        # image = preprocess.tf_equalize_histogram(image)

        return image


    def add_parent_dir(self):
        '''
            Managing the path in the metadata file
        '''
        def add_per_row(row):
            return str(self.data_dir / self.data_types['x']  / Path(row))
        return add_per_row


    def imread(self, img_path):
        '''
            Read Tensorflow image
        '''
        image_string = tf.io.read_file(str(img_path))
        image = tf.image.decode_jpeg(image_string, channels=1)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        return image
