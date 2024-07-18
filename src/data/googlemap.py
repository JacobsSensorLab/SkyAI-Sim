"""
    Load VBN data generated in simulation
    © All rights reserved.
    author: spdkh
    date: June 2023, JacobsSensorLab
"""
import os
import glob
from pathlib import Path
from matplotlib.streamplot import OutOfBounds
from tqdm import tqdm
import numpy as np
import pandas as pd
from natsort import natsorted
import skimage.measure
import tensorflow as tf
import datetime
from types import SimpleNamespace
import pprint
import json


from src.utils import io_helper, geo_helper, preprocess
from src.utils.io_helper import pretty
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

        # Log File

        self.log = SimpleNamespace()
        self.log.args = vars(self.args)

        # Format current date and time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create the new filename with the timestamp
        self.log.filename = f'log_{timestamp}.txt'

        log_features = ['top_left',
                        'bottom_right',
                        'center',
                        'map_size',
                        'single_img_size',
                        'n_raster_imgs']

        for feature in log_features:
            setattr(self.log, feature, SimpleNamespace())

        self.data_info = {'x': 'raster_images'}

        self.input_dim = kwargs['args'].img_size

        self.map_type  = kwargs['map_type'] if 'map_type' in kwargs\
            else 'satellite'
        self.overlap = kwargs['overlap'] if 'overlap' in kwargs\
            else kwargs['args'].overlap

        # This is helpful if we want multiple Googlemap objects in the same program
        data_dir = kwargs['data_dir'] if 'data_dir' in kwargs\
            else kwargs['args'].data_dir

        # Add dataset information to the name of the dataset folder
        data_folder_name = self.map_type + '_' + str(self.overlap)
        self.data_dir = Path(data_dir) / data_folder_name

        img_size = np.round(
                        geo_helper.get_map_dim_m(
                        self.args.fov,
                        self.args.coords[-1],
                        self.args.aspect_ratio[0] / self.args.aspect_ratio[1]
                        ),
                        2
                    )
        self.assign_log('single_img_size',
                        ['x_m', 'y_m', 'z_m'],
                        img_size)

        self.log.single_img_size.area_m2 = \
            round( self.log.single_img_size.x_m *\
                self.log.single_img_size.y_m, 3)

        for i, attr in enumerate(['lat', 'lon']):
            setattr(self.log.top_left, attr, self.args.coords[i])
            setattr(self.log.bottom_right, attr, self.args.coords[i + 2])

        io_helper.check_folder(self.data_dir)

        self.input_dir = None

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
            geo_helper.get_zoom_from_bounds(top_left,
                                            bottom_right)

        self.assign_log('map_size',
                        ['x_pixels', 'y_pixels', 'zoom'],
                        map_size + [map_zoom])

        self.log.center.lat = (top_left[0] + bottom_right[0]) / 2
        self.log.center.lon = (top_left[1] + bottom_right[1]) / 2

        utm = geo_helper.get_utm_epsg((self.log.center.lat, self.log.center.lon))
        if self.args.utm != utm:
            self.args.utm = utm
            pretty('Changing UTM Zone to', utm, log=self, header='Warning!')
        ## Convert geolocation of the raster corners to utm
        # TL and BR points
        self.assign_log('top_left', ['x_utm', 'y_utm'],
                        np.round(geo_helper.geo2utm(top_left[0], top_left[1], self.args.utm), 3))
        self.assign_log('bottom_right', ['x_utm', 'y_utm'],
                        np.round(geo_helper.geo2utm(bottom_right[0], bottom_right[1], self.args.utm), 3))

        ## Convert geolocation of the raster corners to utm
        # Top Left point in meters
        map_tlm = geo_helper.geo2utm(top_left[0], top_left[1], self.args.utm)

        # Bottom Right point in meters
        map_brm = geo_helper.geo2utm(bottom_right[0], bottom_right[1], self.args.utm)

        self.assign_log('map_size', ['x_m', 'y_m'],
                        np.abs(np.subtract(map_tlm[:2], map_brm[:2])).round(3))

        self.log.map_size.area_m2 =\
            round(self.log.map_size.x_m * self.log.map_size.y_m, 3)

        for attr in ['x', 'y']:
             setattr(self.log.n_raster_imgs, attr,
                     int((getattr(self.log.map_size, attr + '_m') \
                         - getattr(self.log.single_img_size, attr + '_m')) /\
                         (getattr(self.log.single_img_size, attr + '_m') * (1 - self.overlap))) + 1
                     )

        self.log.n_raster_imgs.total = \
            self.log.n_raster_imgs.x *\
                self.log.n_raster_imgs.y

        map_name = '_'.join(
            list(map(str, self.args.coords))
            ) + '_' \
                + str(self.args.fov) + '_' \
                    + str(self.args.aspect_ratio[0]) + '_'\
                        + str(self.args.aspect_ratio[1]) + ".jpg"
        map_response = geo_helper.init_static_map(
            [self.log.center.lat, self.log.center.lon],
            self.map_type,
            map_zoom,
            map_size
        )

        tl, br = geo_helper.meters2geo(
            center=self.args.coords[:2],
            img_size=[self.log.single_img_size.x_m,
                      self.log.single_img_size.y_m],
            epsg=self.args.utm)
        raster_zoom, im_size = geo_helper.get_zoom_from_bounds(tl, br)
        self.assign_log('single_img_size',
                ['x_pixels', 'y_pixels', 'zoom'],
                im_size + [raster_zoom])

        io_helper.check_folder(self.data_dir / self.data_info['x'])

        self.log.map_url = map_response.url
        pretty('Data detailed values before download:')
        pprint.pp(self.log)
        # Save the log file with the new filename
        # self.log.T.to_csv()
        io_helper.check_folder(self.data_dir / 'logs')
        with open(self.data_dir / 'logs' / self.log.filename, 'w') as file:
            # Save the namespace to the file
            io_helper.save_namespace(self.log, file)

        if map_name not in os.listdir(self.data_dir):
            geo_helper.get_static_map_image(
                map_response,
                self.data_dir / map_name,
                )
        else:
            pretty('Map image is available in', self.data_dir, 'as', map_name,
                   log=self)

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

        self.check_data()
        self.input_dir = io_helper.find_files(self.data_dir /
                                                 self.data_info['x'],
                                                 'jpg')
        if download_raster:
            ## Start Raster from TL of the map
            # get coordinates of the corners
            # of the top most left image in the raster mission
            tl, br = geo_helper.meters2geo(
                center=self.args.coords[:2],
                img_size=[self.log.single_img_size.x_m,
                          self.log.single_img_size.y_m])
            raster_zoom, im_size = geo_helper.get_zoom_from_bounds(tl, br)
            self.assign_log('single_img_size',
                ['x_pixels', 'y_pixels', 'zoom'],
                im_size + [raster_zoom])

            io_helper.check_folder(self.data_dir / self.data_info['x'])
            self.complete_download()

        self.input_dir = io_helper.find_files(self.data_dir /
                                                 self.data_info['x'],
                                                 'jpg')
        super().config()


    def complete_download(self):
        """
        Finds the latest file downloaded from the data directory
        """
        sorted_imgs = natsorted(
            glob.glob(
                os.path.join(self.data_dir / self.data_info['x'], "*.jpg")
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
        io_helper.wait_for_files(self.input_dir)

        if os.path.exists(self.data_dir / 'meta_data.csv'):
            pretty('Found the metadata file...',
               log=self)
            self.meta_df = pd.read_csv(self.data_dir / 'meta_data.csv')
        else:
            pretty('Generating the metadata file...',
               log=self)
            meta_data = []

            meta_data = [[os.path.basename(path_)] \
                + list(map(
                    float,
                    os.path.basename(path_).split('.jpg')[0].split('_'))) for path_ in self.input_dir
                ]
            self.meta_df = pd.DataFrame(
                meta_data,
                columns=['img_names', 'columns', 'row', 'Lat', 'Lon', 'Alt']
            )
            self.meta_df['entropies'] = self.meta_df.apply(
                self.calc_entropy(self.data_dir / self.data_info['x']), axis=1
            )
            self.meta_df.to_csv(self.data_dir / 'meta_data.csv')

        pretty('All metadata:\n', self.meta_df,
               log=self)

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
                        last_img_name: str = -10):

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
        Returns:
            - None: The function does not return any value but downloads the raster images to the specified directory.
        Example:
            - gen_raster_from_map((34.052235, -118.243683), (34.040713, -118.246769))
        """
        ## Convert geolocation of the raster corners to utm
        # TL point
        map_tlm = geo_helper.geo2utm(top_left_coords[0], top_left_coords[1], self.args.utm)

        # TL point
        map_brm = geo_helper.geo2utm(bottom_right_coords[0], bottom_right_coords[1], self.args.utm)

        ## Start Raster from TL of the map
        # get coordinates of the corners
        # of the top most left image in the raster mission
        im_size = np.asanyarray(self.log.single_img_size.x_m,
                                        self.log.single_img_size.y_m)

        tl, br = geo_helper.meters2geo(center=top_left_coords, img_size=im_size)
        # raster_zoom, im_size = geo_helper.get_zoom_from_bounds(tl, br)

        raster_zoom = self.log.single_img_size.zoom
        # Convert coordinates to UTM
        tlm = geo_helper.geo2utm(tl[0], tl[1], self.args.utm)
        brm = geo_helper.geo2utm(br[0], br[1], self.args.utm)
        x_orig = tlm[0]

        # Width of each raster image along x, y in meters
        raster_wy = np.abs(tlm[1] - brm[1])
        raster_wx = np.abs(tlm[0] - brm[0])

        ## Add vertical margins
        # so that the google mark can be removed later after download
        # the same margin value has to be applied to the preprocess_image method
        im_size[1] += int(im_size[1] * self.args.vmargin/100)

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
            x, y = geo_helper.geo2utm(lat_i, lon_j, self.args.utm)

            # last x and y start from 0
            if [last_x + 1, last_y + 1] == [n_images_x, n_images_y]:
                pretty('All data already downloaded.', log=self)
                return

            pretty('Downloading the rest of Google Map images from \nx = ',
                   last_x + 1, ' / ', n_images_x,
                   '\ny = ', last_y + 1, ' / ', n_images_y, log=self)
            # Determine the UTM coordinates and ID of the nex image to download
            if last_x < n_images_x - 1:
                last_x += 1
                x += raster_wx * (100 - overlap) / 100
            else:
                last_x = 0
                last_y += 1
                y -= raster_wy * (100 - overlap) / 100
                x = x_orig
            lat_i, lon_j = geo_helper.utm2geo(x, y, self.args.utm)
        else:
            pretty('Downloading All Google Map images...', log=self)
            last_x, last_y = 0, 0
            lat_i, lon_j = top_left_coords
            x, y = tlm

        i, j = last_x, last_y

        if n_images_x * n_images_y > 5000:
            pretty('Number of images exceed the publishable limit. Read more at:',
                   '\nhttps://about.google/brand-resource-center/products-and-services/geo-guidelines/#required-attribution/',
                   log=self, header='Warning!')
        io_helper.check_folder(self.data_dir / self.data_info['x'])

        pretty('Do you want to proceed? (y/yes):', end=' ',
                   log=self, header='Attention!')
        response = input().strip().lower()
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

                        output_dir = self.data_dir / self.data_info['x'] / out_name
                        img_response = geo_helper.init_static_map(
                            [lat_i, lon_j],
                            map_type=self.map_type,
                            zoom=raster_zoom,
                            size=im_size
                            )
                        geo_helper.get_static_map_image(
                            img_response,
                            output_dir
                            )
                        x += raster_wx * (100 - overlap) / 100
                        _, lon_j = geo_helper.utm2geo(x, y, self.args.utm)

                        pbar.update()
                        i += 1
                    y -= raster_wy * (100 - overlap) / 100
                    x = x_orig
                    lat_i, lon_j = geo_helper.utm2geo(x, y, self.args.utm)
        else:
            pretty("Not Proceeding the download. Continuing without the download...",
                   log=self)

        print('\t Number of rows and columns:', i, j)


    def cleanup_data(self, entropy_thr=2.1):
        """
        filter out unused data using entropy if map type is roadmap.
        """
        road_dir = self.data_dir
        meta_df = self.meta_df
        if self.map_type != 'roadmap':
            road_folder_name = self.data_dir.name.replace(
                self.map_type, 'roadmap'
            )
            road_dir = self.data_dir.parents[0] / road_folder_name
            road_dir = road_dir / 'meta_data.csv'
            if os.path.exists(road_dir):
                meta_df = pd.read_csv(road_dir)
                pretty('Roadmap data available. Cleaning up based on roadmap entropies...')
            else:
                pretty('Roadmap meta data file is not found.',
                        '\nUsing self meta data file to cleanup data. (Not recommended)',
                        log=self, header='Warning!')

        thr_q = meta_df['entropies'] >= entropy_thr

        self.labels = self.meta_df[thr_q][self.labels.columns]
        self.input_dir = list(self.meta_df[thr_q]['img_names'].apply(
            self.add_parent_dir())
                            )
        pretty('Cleanedup Data from threshold', entropy_thr,
            '\n', self.labels.sample(5),
            '\nNumber of Samples:', len(self.input_dir), log=self)


    def preprocess_image(self, image):
        """
        Preprocess an input image by cropping and resizing.
        Parameters:
            - image (tf.Tensor): Input image to be preprocessed as a TensorFlow tensor.
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
            tf.cast(image_shape[1] * self.args.vmargin/200, dtype=tf.int32),
            tf.cast(image_shape[0] * 1, dtype=tf.int32),
            tf.cast(image_shape[1] * 1 - self.args.vmargin/200, dtype=tf.int32)
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
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        return image

    def assign_log(self, feature, log_attrs, values):
        for attr, value in zip(log_attrs, values):
                setattr(getattr(self.log, feature), attr, value)
