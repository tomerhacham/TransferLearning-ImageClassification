import os

import PIL
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tensorflow.keras import layers

img_height, img_width = 160, 160
batch_size = 32


def get_labels_and_images():
    root = ".\\"
    imglabel_map = os.path.join(root, 'imagelabels.mat')
    setid_map = os.path.join(root, 'setid.mat')
    imagelabels = sio.loadmat(imglabel_map)['labels'][0]
    setids = sio.loadmat(setid_map)
    ids = np.concatenate([setids['trnid'][0], setids['valid'][0], setids['tstid'][0]])
    labels = []
    image_path = []
    for i in ids:
        labels.append(int(imagelabels[i - 1]) - 1)
        image_path.append(os.path.join(root, 'jpg', 'image_{:05d}.jpg'.format(i)))
    return image_path, labels


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path, label):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def split_data(ds, suffle=True,val_prop=0.25, test_prop=0.25):
    if suffle:
        ds = ds.shuffle(len(ds), reshuffle_each_iteration=False)

    val_size = int(len(ds) * val_prop)
    test_size = int(len(ds) * test_prop)
    train_ds = list_ds.skip(val_size+test_size)
    val_ds = list_ds.take(val_size)
    test_ds = list_ds.take(test_size)

    return train_ds,val_ds,test_ds

image_path, labels = get_labels_and_images()
list_ds = tf.data.Dataset.from_tensor_slices((image_path, labels))

val_size = int(len(list_ds) * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def prepare(ds, shuffle=False, augment=False):
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(img_height, img_width),
        layers.Rescaling(1. / 255)
    ])
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)

for image, label in train_ds.take(40):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
