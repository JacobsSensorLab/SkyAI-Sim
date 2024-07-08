"""
    Input Image Data Manager
    © All rights reserved.
    author: spdkh
    date: Nov 2023, JacobsSensorLab
"""

from abc import ABC, abstractmethod
from ast import Raise
from natsort import natsorted

from src.utils import consts
from src.utils.beauty import pretty

class ImageData(ABC):
    """
    A class to represent image data and its preprocessing methods.

    Attributes:
        modes (list): A list of modes for the dataset.
        data_info (dict): A dictionary containing information about the data.
        input_dim (None): The input dimensions of the data.
        data_dir (None): The directory containing the data.
        input_dir (None): The input directory for the data.
        meta_df (None): A dataframe containing metadata.
        labels (None): The labels associated with the data.
        dataset (dict): A dictionary to store the dataset.

    Methods:
        __init__(self, **kwargs):
            Initializes the ImageData object with specified modes and attributes.
        config_input(self): Initial configuration for the input data.
        config_output(self): Abstract method to load output data.
        config(self):
            Configures input and output data by calling config_input() and config_output().
        preprocess_image(self, filename): Abstract method to preprocess the image.
        preprocess_label(self, label): Abstract method to preprocess the labels.
        preprocess(self, img, label): Combines preprocessing of the image and its label.
        imread(self, img_path, shape=self.args.img_size[:2]): Abstract method to read the image.
    """
    def __init__(self, **kwargs):
        """
        Initialize a dataset object with specified modes.
        Parameters:
            - kwargs (dict): Optional keyword arguments.
        Returns:
            - None: Does not return anything.
        Processing Logic:
            - Initializes modes, data_info, input_dim.
            - Sets data_dir and input_dir to None.
            - Initializes meta_df, labels, and dataset.
        """
        super().__init__()
        self.args = kwargs['args']
        self.modes = ['train', 'val', 'test']
        self.data_info = {'x': None, 'y': None}
        self.input_dim = None

        self.data_dir = None
        self.input_dir = None

        self.meta_df = None
        self.labels = None
        self.dataset = {}

    def config_input(self):
        """
            Initial configuration for the input data
        """
        pretty(
            'Number of images in the path:',
            len(self.input_dir),
            info=self
            )
        self.input_dir = natsorted(self.input_dir)

    @abstractmethod
    def config_output(self):
        """
        Abstract method to load output data
        """
        Raise(NotImplementedError())

    def config(self):
        """
        Function Purpose:
            Configures input and output data.
        Parameters:
            - self (object): Object containing input and output data.
        Returns:
            - None: Does not return any value.
        Processing Logic:
            - Calls config_input() and config_output().

        """
        self.config_input()
        self.config_output()

    @abstractmethod
    def preprocess_image(self, filename):
        """
        Abstract method to preprocess the image
        """
        Raise(NotImplementedError)

    def preprocess_label(self, label):
        """
        Abstract method to preprocess the labels
        """
        return label

    def preprocess(self, img, label):
        """
        Combines preprocessing of the image and its label
        """
        return self.preprocess_image(img), self.preprocess_label(label)

    @abstractmethod
    def imread(self, img_path, shape=None):
        """
        Abstract method to read the image
        """
        Raise(NotImplementedError())
