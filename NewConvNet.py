import tensorflow as tf
from TF_Helper import *
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('n_classes', 2, 'Number of classes in the data.')
flags.DEFINE_integer('batch_size', 70, 'Mini-Batch Size.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('train_file', 'garage_door224_TRAIN.tfrecords', 'Name of the TFRecords file used for training.')
flags.DEFINE_string('test_file', 'garage_door224_TEST.tfrecords', 'Name of the TFRecords file used for testing.')
flags.DEFINE_string('train_dir','/home/mcamp/PycharmProjects/GarageDoor/',
                    'Path to the directory housing training data and other project files.')
flags.DEFINE_integer('n_epochs', 5000, 'The number of epochs the model is to run.')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

with tf.name_scope('TrainingData'):
    X_train_batch, y_train_batch = inputs(FLAGS.train_dir,
                                          FLAGS.train_file,
                                          FLAGS.batch_size,
                                          FLAGS.n_epochs,
                                          FLAGS.n_classes,
                                          one_hot_labels=True,
                                          imshape=150528)

with tf.name_scope('TestingData'):
    X_test_batch, y_test_batch = inputs(FLAGS.train_dir,
                                        FLAGS.test_file,
                                        75,
                                        FLAGS.n_epochs,
                                        FLAGS.n_classes,
                                        one_hot_labels=True,
                                        imshape=150528)

with tf.Session() as sess:

    with tf.name_scope('SampleImages'):
        S = tf.placeholder(tf.float32, shape=[5, 224, 224, 3])
        # sample_labels = tf.placeholder(tf.float32, shape=[None, 2])


    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, 224 * 224 * 3])
        y_ = tf.placeholder(tf.float32, shape=[None, FLAGS.n_classes])

    with tf.name_scope('Conv1'):
        W_conv1 = weight_variable([5, 5, 3, 25])
        b_conv1 = bias_variable([25])

        X_image = tf.reshape(X, [-1, 224, 224, 3])

        h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('Conv2'):
        W_conv2 = weight_variable([5, 5, 25, 50])
        b_conv2 = bias_variable([50])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('Dense'):
        W_fc1 = weight_variable([56 * 56 * 50, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 56 * 56 * 50])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, FLAGS.n_classes])
        b_fc2 = bias_variable([FLAGS.n_classes])

    with tf.name_scope('Out'):
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('LossFunction'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    tf.summary.scalar('Loss', cross_entropy)
    tf.summary.image("InputImages", X_image)
    tf.summary.image("SampleImages", S)

    with tf.name_scope('Accuracy'):
        with tf.name_scope('CorrectPrediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
        with tf.name_scope('Accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('AccuracyScore', accuracy)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter('./train', sess.graph)
    test_writer = tf.summary.FileWriter('./test', sess.graph)



    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    X_test, y_test = sess.run([X_test_batch, y_test_batch])
    merged = tf.summary.merge_all()
    for epoch in range(FLAGS.n_epochs):
        X_train, y_train = sess.run([X_train_batch, y_train_batch])
        # code to add labels to images for tensorboard
        im_samples = []
        im_labels = []
        for i in range(5):
            img = np.reshape(X_train[i], [224, 224, 3]).astype('uint8')
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default().font
            label = str(y_train[i])
            draw.text((0, 0), label, (255, 255, 255), font=font)

            img = np.reshape(np.asarray(img), [224, 224, 3])

            im_labels.append(y_train[i])
            im_samples.append(img)
        im_samples = np.asarray(im_samples).astype('float32')
        im_labels = np.asarray(im_labels).astype('float32')
        #########################################################
        if epoch%100 == 0:
            summary, train_accuracy = sess.run([merged, accuracy], feed_dict={
                X: X_train, y_: y_train, keep_prob: 1.0, S: im_samples
            })
            train_writer.add_summary(summary, epoch)
            print("Step %d, Training accuracy %g"%(epoch,train_accuracy))
            save_path = saver.save(sess, './model/model.ckpt')
        summary, loss = sess.run([merged, train_step], feed_dict={X: X_train, y_: y_train,
                                                                  keep_prob: FLAGS.dropout, S: im_samples})
        train_writer.add_summary(summary, epoch)
        if epoch%50 == 0:
            summary, test_accuracy = sess.run([merged, accuracy], feed_dict={
                X: X_test, y_: y_test, keep_prob: 1.0, S: im_samples
            })
            print("Test Accuracy %g"% test_accuracy)
            test_writer.add_summary(summary, epoch)

    coord.request_stop()
    coord.join(threads)
