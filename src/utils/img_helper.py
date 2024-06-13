"""
    author: SPDKH
    year: 2023
    caGAN project image edit helper functions
"""
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from src.utils import consts, norm_helper, geo_helper


def choose_random_images(
    n_imgs: int,
    imgs_paths: List[str],
    output_dir: str = 'sample_images',
    obj=None,  # module
    seed: int = None
    ):
    """
    This function randomly selects a specified number of images
    from a given list of image paths and plots them in a grid format.
    Parameters:
        - n_imgs (int): Number of images to be randomly selected.
        - imgs_paths (list): List of image paths: n * x * y (* ch)
        - output_dir (str): Directory to save the plotted images.
                            Default is 'sample_images'.
        - obj (module): Module used to read the images.
                        Default is None.
        - seed (int): Seed for the random selection.
                    Default is None.
    Returns:
        - imgs (list): List of randomly selected images.
        - titles (list): List of titles for each image.
    Processing Logic:
        - Randomly selects images from given paths.
        - Plots images in a grid format.
        - Uses given module to read images.
        - Uses given seed for random selection.

    """
    np.random.seed(seed)
    rand_samples = np.random.choice(len(imgs_paths),
                                    n_imgs, replace=False)
    my_imread = imread if obj is None else obj.imread

    imgs = [my_imread(imgs_paths[idx]) for idx in rand_samples]

    # The title for each image, extracting relevant parts
    titles = ['\n'.join(imgs_paths[idx].split('/')[-1][:-4].split('_')[2:4])\
        for idx in rand_samples]

    plot_multy(imgs, output_dir, n_imgs, 1,
               titles=titles)
    return imgs, titles


def plot_multy(
    imgs: List[np.ndarray],
    output_dir: str,
    cols: int,
    rows: int = 1,
    titles: List[str] = None
    ) -> None:
    """
    Plots multiple images in a grid layout.
    Parameters:
        - imgs (list): List of images to be plotted: n * x * y (* ch)
        - output_dir (str): Directory to save the plotted images.
        - cols (int): Number of columns in the grid layout.
        - rows (int): Number of rows in the grid layout.
                    Default is 1.
        - titles (list): List of titles for each image.
                        Default is None.
    Returns:
        - None: The function does not return any value.
    Processing Logic:
        - Creates a grid layout with specified number of rows and columns.
        - Displays each image in the grid layout.
        - Adds a title to each image if titles are provided.
        - Saves the plotted images in the specified output directory.
        - Closes all figures to avoid memory leak.
        """
    _, ax = plt.subplots(nrows=rows, ncols=cols,
                        figsize=(cols * 4,rows * 4),
                        subplot_kw=dict(xticks=[], yticks=[]))
    if rows == 1:
        ax = np.expand_dims(ax, 0)

    cmap = 'gray' if len(imgs[0].shape) == 2 else None
    # iterate over each row and column in the grid and display an image
    # The images to be displayed are accessed from the variable x
    i = 0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].imshow(imgs[i], cmap=cmap)
            if titles is None:
                ax[row, col].set_title(str(row)+str(col),
                                       fontsize=14)
            else:
                ax[row, col].set_title(titles[i], fontsize=14)
            i += 1

    # Add title to the left of y-axis and rotate it by 90 degrees
    plt.figtext(0.1, 0.5, output_dir.split('/')[-1],
                va='center', ha='center', rotation=90, fontsize=16)

    # show the figure
    plt.show()
    plt.savefig(output_dir)  # Save sample results
    plt.close("all")  # Close figures to avoid memory leak


def make_pairs(
    data, mode: str, overlap=100, return_overlap=False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function:
        Make pairs based on the amount of overlap in one or two dataset/s
        images with location overlap of above overlap% is positive (1)
        images with location overlap of below min(100-overlap, overlap)% is negative (0)
        This is a specific task for the siamese network pairing.
    Parameters:
        - data (tuple | obj): A tuple of objects or and object
            containing data information.
        - mode (str): A string indicating the mode of the data,
            either of the following ['train', 'val', 'test']
        - overlap (int): Desired overlap percentage,
            Default is 100.
        - return_overlap (bool):
            Whether to make the label to be the overlap amount
            rather than 0 or 1
    Returns:
        - pair_images (np.array): A 2-tuple of image pairs.
        - pair_labels (np.array):
            A 2-tuple of labels for the image pairs
            if return overlap is enabled: overlap percentage
            o.w. (0 or 1) for each pair.
    Processing Logic:
        - Set random seed using data1's seed.
        - Retrieve images and labels from data1 and data2.
        - Calculate bounding boxes for labels.
        - Loop through all images.
        - Randomly select an image with the same class label.
        - Calculate bounding boxes for labels.
        - Check if bounding boxes overlap.
        - If not, repeat until they do.
        - Add positive and negative image pairs and labels to lists.
        - Return a 2-tuple of image pairs and labels.

    Cautious: Overlapping training dataset might lead to training pollution
    todo: mix it with make_triplets function
    """
    if isinstance(data, tuple):
        data1, data2 = data
    else:
        data1 = data2 = data

    np.random.seed(data1.args.seed)
    images = data1.data_info['x' + mode]
    labels = np.asarray(data1.data_info['y' + mode])

    images2 = data2.data_info['x' + mode]
    labels2 = np.asarray(data2.data_info['y' + mode])

    ### extract images and labels that are present in both datasets
    # and raise an error if there is not much present in both
    if isinstance(data, tuple):
        if not np.array_equal(labels, labels2):
            idx_a = labels == labels2
            images = images[idx_a]
            images2 = images2[idx_a]
            labels = labels[idx_a]

        if len(labels) < 0.9 * np.max(len(labels2), len(labels)):
            raise ValueError("Data misalignment. Losing more than 10% of the original data.")

    cols = data1.data_info['ytrain'].columns
    pair_images = []
    pair_labels = []
    # loop over all images
    with tqdm(position=0, leave=True, total=len(images)) as pbar:
        for idx_a, cur_img in enumerate(images):
            # take the actual coordinates of the current image location
            label = norm_helper.norm_undo(labels[idx_a],
                                          data1.org_out_max[cols],
                                          data1.org_out_min[cols])

            # calculate top left and bottom right coordinates for current image
            coords = geo_helper.calculate_bounding_box(
                label[0: 2],
                label[2],
                data1.dim['input']
            )

            # Find a random sample coordinates
            coords_b = find_random_sample(
                            range(len(labels)),
                            labels, data2
            )

            ### binarize based on overlap, if binary labels are requested
            if not return_overlap:
                overlap_flag = geo_helper.overlapped(coords,
                                            coords_b,
                                            overlap)
                ## if current image is overlapped with the random image,
                # or there are two datasets we want to fix the positive label,
                # then search for the negative label
                if overlap_flag or isinstance(data, tuple):
                    # if there are two dataset,
                    # the positive label is shared between the two datasets
                    if isinstance(data, tuple):
                        pos_img = images2[idx_a]
                    # if there is only one dataset,
                    # the positive label is the randomly selected label with overlap
                    else:
                        pos_img = images2[idx_b]

                    ## look to find random coordinates with no overlap
                    # for the negative label
                    while geo_helper.overlapped(
                        coords, coords_b,
                        overlap=np.min(100-overlap, overlap)
                    ):
                        idx_b, label_b, coords_b = find_random_sample(
                            range(len(labels)),
                            labels, data2
                        )

                    neg_img = images2[idx_b]
                ## if current image is overlapped with the random image,
                # we can fix the negative label,
                # then search for the positive among the ones with overlap
                else:
                    neg_img = images2[idx_b]

                    while not geo_helper.overlapped(coords,
                                                coords_b,
                                                overlap=overlap):
                        idx_b, label_b, coords_b = find_random_sample(
                            range(len(labels)),
                            labels, data2
                        )
                        idx_b = np.random.choice(range(len(images)))

                    pos_img = images2[idx_b]

                # prepare a positive pair of images and update our lists
                pair_images.append([cur_img, pos_img])
                pair_labels.append([1])

                # prepare a negative pair of images and update our lists
                pair_images.append([cur_img, neg_img])
                pair_labels.append([0])
            else:
                pair_images.append([cur_img, images2[label_b]])
                pair_labels.append(
                    [geo_helper.find_overlap(coords,
                                             coords_b)]
                )
                pbar.update()
        # return a 2-tuple of our image pairs and labels
    return np.array(pair_images), np.array(pair_labels)


def find_random_sample(
    n_images: int, labels: np.ndarray, data: object
    ) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Finds a random sample from a dataset and returns its index, label, and coordinates.
    Parameters:
        - n_images (int): Number of images in the dataset.
        - labels (array): Array of labels for each image.
        - data (object): Object containing information about the dataset.
    Returns:
        - idx (int): Index of the randomly selected image.
        - label (array): Label of the randomly selected image.
        - coords (array): Coordinates of the bounding box for the label.
    Processing Logic:
        - Choose random index from dataset.
        - Normalize label values.
        - Calculate bounding box coordinates.
        - Return index, label, and coordinates.

    """
    idx = np.random.choice(n_images)
    cols = data.data_info['ytrain'].columns
    label = norm_helper.norm_undo(
        labels[idx],
        data.org_out_max[cols],
        data.org_out_min[cols]
    )
    coords = geo_helper.calculate_bounding_box(
        label[:2],
        label[2],
        data.dim['input']
    )
    return idx, label, coords


def make_triplets(data1, data2, mode):
    """
        Make triplets based on the images from two different datasets
        the first data indicates the anchor image,
        the image from the same location but different dataset is positive
        the image from a different location and a different dataset is negative

        params:
            data1: data object
            data2: data object
            mode: either of the following ['train', 'val', 'test']

        returns:
            a keras dataset, eache element includes three images:
            0. anchor
            1. positive
            2. negative
    """
    np.random.seed(data1.args.seed)
    images = data1.data_info['x' + mode]
    labels = np.asarray(data1.data_info['y' + mode])
    cols = data1.data_info['y' + mode].columns

    images2 = data2.data_info['x' + mode]
    labels2 = np.asarray(data2.data_info['y' + mode])

    ### extract images and labels that are present in both datasets
    # and raise an error if there is not much present in both
    if not np.array_equal(labels, labels2):
        idx_a = labels == labels2
        images = images[idx_a]
        images2 = images2[idx_a]
        labels = labels[idx_a]

    if len(labels) < 0.9 * np.max(len(labels2), len(labels)):
        raise ValueError("Data misalignment. Losing more than 10% of the original data.")

    pos_imgs = []
    neg_imgs = []
    # loop over all images
    with tqdm(position=0, leave=True, total=len(images)) as pbar:
        for idx_a, _ in enumerate(images):
            # grab the current image and label belonging to the current iteration
            label = norm_helper.norm_undo(labels[idx_a],
                                            data1.org_out_max[cols],
                                            data1.org_out_min[cols])

            pos_img = images2[idx_a]
            pos_imgs.append(pos_img)
            coords = geo_helper.calculate_bounding_box(label[0: 2],
                                                        label[2],
                                                        data1.dim['input'])
            idx_b, _, coords_b = find_random_sample(
                            range(len(labels)),
                            labels, data2
                        )

            while geo_helper.overlapped(coords, coords_b, 0.2):
                idx_b, _, coords_b = find_random_sample(
                            range(len(labels)),
                            labels, data2
                        )
            # grab the indices for each of the class labels *not* equal to
            # the current label and randomly pick an image corresponding
            # to a label *not* equal to the current label
            neg_img = images2[idx_b]
            neg_imgs.append(neg_img)

            pbar.update()
    # return a 2-tuple of our image pairs and labels

    images = tf.data.Dataset.from_tensor_slices(images)
    pos_imgs = tf.data.Dataset.from_tensor_slices(pos_imgs)
    neg_imgs = tf.data.Dataset.from_tensor_slices(neg_imgs)
    dataset = tf.data.Dataset.zip((images,
                                    pos_imgs,
                                    neg_imgs))

    return dataset


def img_batch_load(
    data,
    imgs_paths: List[str],
    batch_size: int,
    iteration: int = 0
) -> Dict[str, np.ndarray]:
    """
    Parameters
    ----------
    path: str

    iteration: int
        batch iteration id to load the right batch
        pass batch_iterator(.) directly if loading batches
        this updates the batch id,
        then passes the updated value
        can leave 0 if
    batch_size: int
        if not loading batches,
        keep it the same as number of all samples loading

    Returns: array
        loaded batch of raw images
    -------
    """
    imgs_paths.sort()

    iteration = iteration * batch_size
    imgs_paths = imgs_paths[iteration:batch_size + iteration]

    image_batch = {}
    for i, path in enumerate(imgs_paths):
        cur_img = data.imread(imgs_paths[i])
        image_batch[path] = cur_img.copy()
    return image_batch


def pair_batch_load(data, pairs, batch_size, iteration):
    """
        Parameters
        ----------
        pairs: tuple of two str paths

        iteration: int
            batch iteration id to load the right batch
            pass batch_iterator(.) directly if loading batches
            this updates the batch id,
            then passes the updated value
            can leave 0 if
        batch_size: int
            if not loading batches,
            keep it the same as number of all samples loading

        Returns: array
            loaded batch of raw images
        -------
        """
    iteration = iteration * batch_size
    image_batch = []
    for i in range(batch_size):
        pair = pairs[i + iteration]
        image_batch.append([])
        # pretty('Empty batch:', image_batch[i])
        for j in range(2):
            img = data.imread(pair[j])
            image_batch[i].append(img)
    return image_batch

def imread(img_path, shape=consts.IMG_SIZE[:2]):
    """
    Reads and normalizes images with specified dimension.
    Parameters:
        - img_path (string): Path to the image file.
        - shape (2D tuple of int): Desired shape of the image. Defaults to IMG_SIZE[:2].
    Returns:
        - img (type): Normalized image as a numpy array.
    Processing Logic:
        - Convert image to RGB if not already.
        - Resize image to specified shape.
        - Normalize image using min-max normalization.
    """
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        print(np.shape(img))
        img = img.resize(shape)
        img = norm_helper.min_max_norm(np.asarray(img))
    return img

def save_sample_data(data,
                    sample_label,
                    sample_img_name,
                    n_sample_imgs = 2):

    """
    Save sample data and corresponding labels as images for visualization purposes."
    Parameters:
        - data (list): List of sample data to be saved.
        - sample_label (list): List of labels corresponding to the sample data.
        - sample_img_name (str): Name of the sample image.
        - n_sample_imgs (int): Number of sample images to be saved. Default is 2.
    Returns:
        - None.
    Processing Logic:
        - Create a unique name for each sample image.
        - Convert sample data to numpy array.
        - Plot and save the sample image.
        - Repeat for specified number of sample images.
    """
    for i in range(n_sample_imgs):
        label = sample_label[i]
        sample_name = sample_img_name

        sample_name += '_' + str(i) + '_' + str(label) + '.png'
        sample_dir = consts.SAMPLE_DIR / sample_name

        sample_img = np.asarray(data[i])
        plt.figure()
        plt.imshow(sample_img)
        plt.title(sample_img_name + ' ' + str(i) + ' ' + str(label))
        plt.savefig(sample_dir)  # Save sample results
