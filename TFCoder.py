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

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#takes a csv file in the format of: path,label
def read_qfile(qfile):
    q = pd.read_csv(qfile, header = None)
    q.columns = ['path', 'label']
    labels = q.label.tolist()
    path = q.path.tolist()
    return path, labels

'''takes the list of plane text labels (ie Open, Closed) and converts to 1 and 0.
this will return an array of all the labels.
Example:
labels = ['open','open','open','closed','closed','open','open','closed']
int_classes = [0,0,0,1,1,0,0,1]
index = ['open', 'closed']'''
def label_to_int(labels=None, index=None): #labels is a list of labels
    class_index = index
    int_classes = []
    for label in labels:
      int_classes.append(class_index.index(label)) #the class_index.index() looks values up in the list label
    int_classes = np.array(int_classes, dtype=np.uint32)
    return int_classes

#just a helper function to read in all the images to an array
def read_images(pathlist=None):
    images = []
    labels = []
    for file in pathlist:
        im = Image.open(file)
        im = np.asarray(im, np.uint8)
        image_name = file.split('/')[-1].split('.')[0]
        if "open" in image_name:
            image_name = 1
        elif "closed" in image_name:
            image_name = 0
        else:
            image_name = 99
        images.append([image_name,im])
    images = sorted(images, key = lambda image: image[0])
    images_only = [np.asarray(image[1].flatten(), np.uint8) for image in images]
    images_only = np.array(images_only)
    labels_only = [np.asarray(key[0], np.int32) for key in images]
    labels_only = np.array(labels_only)
    # print(images_only.shape)
    # print(labels.shape)
    return images_only, labels_only #flat images


def convert_to_TF(images, labels, name):
    label_count = labels.shape[0]
    print('There are %d images in this dataset.' % (label_count))
    if images.shape[0] != label_count:
        raise ValueError('WTF! Devil! There are %d images and %d labels. Go fix yourself!' %
                         (images.shape[0], label_count))
    # rows = images.shape[1]
    # cols = images.shape[2]
    # depth = images.shape[3]
    rows = 100
    cols = 100
    depth = 3

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



path, labels = read_qfile('queue.txt')
X_train, X_test, y_train, y_test = train_test_split(
    path, labels, test_size=0.10, random_state=42)

label_index = ['open', 'closed']
# y_train = label_to_int(y_train, label_index)
# y_test = label_to_int(y_test, label_index)
X_train, y_train = read_images(pathlist=X_train)
X_test, y_test = read_images(pathlist=X_test)
num = random.randint(0,8000)
print(y_train[num])
sample_im = np.reshape(X_train[num], [224, 224, 3])
plt.imshow(np.asarray(sample_im, np.uint8))
plt.show()
convert_to_TF(X_train, y_train, 'garage_door224_TRAIN')
convert_to_TF(X_test, y_test, 'garage_door224_TEST')

train_size = (8243, 30000)

