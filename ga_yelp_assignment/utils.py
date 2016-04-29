import csv
import numpy as np
import shutil
import os
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
from skimage.color import rgb2gray
from tensorflow.contrib import skflow
from flask import current_app


try:
    try:
        from scipy.misc import imread
    except ImportError:
        from scipy.misc.pilutil import imread
    from scipy.misc import imresize
except ImportError:
    raise ImportError("The Python Imaging Library (PIL)"
                      " is required to load data from jpeg files")


def _load_csv_into_dict(csv_loc):
    data = OrderedDict()
    with open(csv_loc, 'rU') as csvfile:
        rows = csv.reader(csvfile)
        # Skip the first row
        next(rows)
        for row in rows:
            if row[0] in data:
                raise Exception("Duplicate id found!")
            data[row[0]] = row[1]
    return data


def create_photo_label_dict(labels_file, photo_file, photos_dir, data_id='0'):
    label_dict = _load_csv_into_dict(labels_file)
    photo_dict = _load_csv_into_dict(photo_file)
    retval = {}
    for k, v in photo_dict.items():
        if k in retval:
            raise Exception("Duplicate id found!")
        loc = os.path.join(photos_dir, "{}.jpg".format(k))
        retval[loc] = 1 if data_id in label_dict.get(v, '').split() else 0
    return retval


def _remove_files_from_directory(dir_):
    for fle in os.listdir(dir_):
        fp = os.path.join(dir_, fle)
        if os.path.isfile(fp):
            os.unlink(fp)


def save_model(classifier, dir_):
    _remove_files_from_directory(dir_)
    classifier.save(dir_)


def load_model(dir_):
    return skflow.TensorFlowEstimator.restore(dir_)


def convert_rgb_to_greyscale(rgb_img):
    # Convert rgb to grey scale, 3d matrix to 2d matrix 
    return rgb2gray(rgb_img)


def resize_image(img):
    # Image must be 2d
    return imresize(img, (
        current_app.config.get('IMAGE_X_DIM'),
        current_app.config.get('IMAGE_Y_DIM')))


def flatten_image(img):
    # Flatten the image into one array of pixels
    return img.reshape(1, img.shape[0] * img.shape[1])


def lazy_load_files(file_name_value_dict, chunk_size=100):
    for i in range(0, len(file_name_value_dict), chunk_size):
        data = []
        values = []
        chunk_file_name_list = list(file_name_value_dict.keys())[i: i + chunk_size]
        for path_name in chunk_file_name_list:
            img = imread(path_name)
            if img.ndim is 0:
                raise Exception(
                    "Failed to read the image file %s, "
                    "Please make sure that libjpeg is installed"
                    % path_name)
            # Convert the image to greyscale
            gs_img = convert_rgb_to_greyscale(img)
            # Resize the image, all images have to be same size
            gs_img = resize_image(gs_img)
            # Flatten features to 1d array
            wide_img = flatten_image(gs_img)
            pixels = np.asarray(wide_img[0], dtype=np.float32)
            data.append(pixels)
            values.append(file_name_value_dict[path_name])
        yield np.array(data), np.array(values), chunk_file_name_list


def get_set_of_image_sizes(container_path):
    # Caution this method can take a while depending on how
    # big the image directory is.
    sizes = {}
    file_name_list = [f for f in listdir(container_path) if not f.startswith('.')]
    for name in file_name_list:
        path_name = os.path.join(container_path, name)
        img = imread(path_name)
        pixels = np.asarray(img, dtype=np.float32)
        x, y, z = pixels.shape
        tupe = (x, y)
        if tupe not in sizes:
            sizes[tupe] = 0
        sizes[tupe] += 1
    return sizes


def get_label_counts(labels_file):
    label_dict = _load_csv_into_dict(labels_file)
    label_count = {}
    for k, v in label_dict.items():
        biz_labels = v.split()
        for lab in biz_labels:
            if lab not in label_count:
                label_count[lab] = 0
            label_count[lab] += 1
    return label_count
