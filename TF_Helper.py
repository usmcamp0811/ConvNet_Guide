import tensorflow as tf
import os


def read_and_decode(filename_queue,imshape=50176):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([imshape])
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.cast(image, tf.float32)

    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(train_dir, file, batch_size, num_epochs, n_classes, one_hot_labels=False, imshape=50176):

    if not num_epochs: num_epochs = None
    filename = os.path.join(train_dir, file)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue, imshape)

        if one_hot_labels:
            label = tf.one_hot(label, n_classes, dtype=tf.int32)

        example_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=1,
            capacity=1000, enqueue_many=False,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=10, name=file)

    return example_batch, label_batch

def inputs2(train_dir, file, batch_size, num_epochs, n_classes, one_hot_labels=False, imshape=50176):

    if not num_epochs: num_epochs = None
    filename = os.path.join(train_dir, file)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue, imshape)

        if one_hot_labels:
            label = tf.one_hot(label, n_classes, dtype=tf.int32)

        example_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=1,
            capacity=1000, enqueue_many=False,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=10, name=file)

    return example_batch, label_batch