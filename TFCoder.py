import pandas as pd
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import PIL
from PIL import Image, ImageOps
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_q_list(filepathlist):
    filepathlist = filepathlist
    filepaths = []
    labels = []
    for path in filepathlist:
        image_files = os.listdir(path)
        for image in image_files:
            if image.endswith('.jpg'):
                image_file = os.path.join(path, image)
                image_file = image_file
                image_label = os.path.basename(os.path.normpath(path))
                filepaths.append(image_file)
                labels.append(image_label)

    return filepaths, labels

def label_to_int(labels=None, index=None): #labels is a list of labels
    class_index = index
    int_classes = []
    for label in labels:
        int_classes.append(class_index.index(label)) #the class_index.index() looks values up in the list label
    int_classes = np.array(int_classes, dtype=np.uint32)
    return int_classes

def read_images(pathlist, lbl_index_list, size=None):
    images = []
    labels = []
    for file in tqdm(pathlist):
        im = Image.open(file)
        im = np.asarray(im, np.uint8)
        if size != None:
            im = image_resizer(im, size)
        image_label_text = [os.path.basename(os.path.dirname((file)))]
        image_label_int = label_to_int(labels=image_label_text, index=lbl_index_list)
        images.append([image_label_int,im])
    images = sorted(images, key = lambda image: image[0])
    image_shape = images[0][1].shape
    images_only = [np.asarray(image[1].flatten(), np.uint8) for image in images]
    images_only = np.array(images_only)
    labels_only = [np.asarray(key[0], np.int32) for key in images]
    labels_only = np.array(labels_only)

    return images_only, labels_only, image_shape #flat images

def convert_to_TF(images, labels, image_shape, name):
    label_count = labels.shape[0]
    print('There are %d images in this dataset.' % (label_count))
    if images.shape[0] != label_count:
        raise ValueError('WTF! Devil! There are %d images and %d labels. Go fix yourself!' %
                         (images.shape[0], label_count))
    rows = image_shape[0]
    cols = image_shape[1]
    depth = image_shape[2]

    filename = os.path.join(name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(label_count):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

def image_resizer(image,size):

    basewidth = size
    img = Image.fromarray(image)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    #crop the sides off to make a square image
    img = ImageOps.fit(img, (size,size), Image.ANTIALIAS)

    return np.asarray(img)

def label_on_image(image, image_label_text):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.load_default().font
    font = ImageFont.truetype("arial.ttf", 45)
    label = image_label_text


    draw.text((10, 10), str(image_label_text), (255, 0, 0), font=font)
    # img = np.reshape(np.asarray(img), [160, 160, 3])

    return np.asarray(img)
