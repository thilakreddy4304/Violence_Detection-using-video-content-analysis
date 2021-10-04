
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import sys
import h5py

import download

from random import shuffle


import tensorflow as tf
print(tf.__version__)


img_size = 224
img_size_touple = (img_size, img_size)

num_channels = 3

img_size_flat = img_size * img_size * num_channels

num_classes = 2

_num_files_train = 1

_images_per_file = 20

_num_images_train = _num_files_train * _images_per_file

video_exts = ".mp4"

in_dir = "video"
in_dir_prueba = 'video'


def print_progress(count, max_count):

    pct_complete = count / max_count

    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def get_frames(current_dir, file_name):
    in_file = os.path.join(current_dir, file_name)

    images = []

    vidcap = cv2.VideoCapture(in_file)

    success, image = vidcap.read()

    count = 0

    while count < _images_per_file:

        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res = cv2.resize(RGB_img, dsize=(img_size, img_size),
                         interpolation=cv2.INTER_CUBIC)

        images.append(res)

        success, image = vidcap.read()

        count += 1

    resul = np.array(images)

    resul = (resul / 255.).astype(np.float16)

    return resul


image_model = ResNet50(include_top=True, weights='imagenet')
image_model1 = VGG16(include_top=True, weights='imagenet')
image_model1.summary()


transfer_layer = image_model1.get_layer('fc2')
image_model1_transfer = Model(inputs=image_model1.input,
                              outputs=transfer_layer.output)
transfer_values_size = K.int_shape(transfer_layer.output)[1]
print("La entrada de la red dimensiones:",
      K.int_shape(image_model1.input)[1:3])
print("La salida de la red dimensiones: ", transfer_values_size)


def get_transfer_values(current_dir, file_name):

    shape = (_images_per_file,) + img_size_touple + (3,)

    image_batch = np.zeros(shape=shape, dtype=np.float16)

    image_batch = get_frames(current_dir, file_name)

    shape = (_images_per_file, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    transfer_values = \
        image_model1_transfer.predict(image_batch)

    return transfer_values


def proces_transfer(vid_names, in_dir, labels):
    count = 0

    tam = len(vid_names)

    shape = (_images_per_file,) + img_size_touple + (3,)

    while count < tam:
        video_name = vid_names[count]

        image_batch = np.zeros(shape=shape, dtype=np.float16)

        image_batch = get_frames(in_dir, video_name)

        shape = (_images_per_file, transfer_values_size)
        transfer_values = np.zeros(shape=shape, dtype=np.float16)

        transfer_values = \
            image_model1_transfer.predict(image_batch)

        labels1 = labels[count]

        aux = np.ones([20, 2])

        labelss = labels1 * aux

        yield transfer_values, labelss

        count += 1


def make_files(n_files, names_training, in_dir_prueba, labels_training):
    gen = proces_transfer(names_training, in_dir_prueba, labels_training)
    numer = 1

    chunk = next(gen)
    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    with h5py.File('prueba.h5', 'w') as f:

        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)

        dset[:] = chunk[0]
        dset2[:] = chunk[1]
        for chunk in gen:
            if numer == n_files:
                break

            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            print_progress(numer, n_files)
            numer += 1


def make_files_validation(n_files, names_validation, in_dir_prueba, labels_validation):
    gen = proces_transfer(names_validation, in_dir_prueba, labels_validation)
    numer = 1

    chunk = next(gen)
    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]

    with h5py.File('pruebavalidation.h5', 'w') as f:

        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)

        dset[:] = chunk[0]
        dset2[:] = chunk[1]
        for chunk in gen:
            if numer == n_files:
                break

            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            print_progress(numer, n_files)
            numer += 1


def label_video_names(in_dir):
    names = []
    labels = []
    for current_dir, dir_names, file_names in os.walk(in_dir):
        for file_name in file_names:
            if file_name[0:2] == 'vi':
                labels.append([1, 0])
                names.append(file_name)
            elif file_name[0:2] == 'no':
                labels.append([0, 1])
                names.append(file_name)
    c = list(zip(names, labels))
    shuffle(c)
    names, labels = zip(*c)
    return names, labels


def process_alldata_training():
    joint_transfer = []
    frames_num = 20
    count = 0

    with h5py.File('prueba.h5', 'r') as f:

        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch) / frames_num)):
        inc = count + frames_num
        joint_transfer.append([X_batch[count:inc], y_batch[count]])
        count = inc

    data = []
    target = []

    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))

    return data, target


def process_alldata_validation():
    joint_transfer = []
    frames_num = 20
    count = 0

    with h5py.File('pruebavalidation.h5', 'r') as f:

        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch) / frames_num)):
        inc = count + frames_num
        joint_transfer.append([X_batch[count:inc], y_batch[count]])
        count = inc

    data = []
    target = []

    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))

    return data, target


def main():

    current_dir = os.path.dirname(os.path.abspath(
        __file__))  # absolute path of current directory

    current_dir = os.path.dirname(os.path.abspath(__file__))
    trained_model_path = os.path.join(current_dir, 'trained_net.h5')

    names, labels = label_video_names(in_dir_prueba)
    print("names", names)

    validation_set = int(len(names))
    names_validation = names
    labels_validation = labels

    make_files_validation(validation_set, names_validation,
                          in_dir_prueba, labels_validation)

    data_val, target_val = process_alldata_validation()

    model = load_model(trained_model_path)
    prediction = model.predict(np.array(data_val))
    print("\n prediction \n", prediction)

    for idx in range(len(prediction)):
        print("Amount of Violence in %s is %f %%" %
              (names[idx], prediction[idx, 0]*100))


if __name__ == '__main__':

    main()
