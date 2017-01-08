import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from TFCoder import *
from TF_Helper import inputs
import numpy as np


class BasicConvNet(object):

    def __init__(self, flat_input_image_size, num_classes, network_architecture,
                 transfer_fct=tf.nn.softmax_cross_entropy_with_logits,
                 learning_rate=1e-4, batch_size=75, imshape=(160,160,3), dropout=0.5, model_dir='./'):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.imshape = imshape
        self.reshape = [-1, imshape[0], imshape[1], imshape[2]]
        self.dropout = dropout
        self.model_dir = model_dir
        self.flat_input_image_size = flat_input_image_size
        #Graph inputs
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.flat_input_image_size])
            self.y_ = tf.placeholder(tf.float32, shape=[None, num_classes])

        self._create_network()
        self._create_loss_optimizer()

        #initialize tf variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


        #TODO: Add model save and load feature somewhere in this area
        #launch a session
        self.sess = tf.Session()
        self.sess.run(init_op)
        self.saver = tf.train.Saver(max_to_keep=3)
        self.ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        self.merged = tf.summary.merge_all()
        #TODO: Maybe reshape X here.. not sure yet..
        #TODO: allow for the ability to change summary writer dir
        self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter('./test', self.sess.graph)
    def _create_network(self):
        #initialize network weights and biases
        self.network_weights = self._initialize_weights(**self.network_architecture)
        #TODO: Allow for the number of convolutions to be controlled
        #reshape the flat image back to its original shape
        X_image = tf.reshape(self.x, self.reshape)
        self.keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('Conv1'):
            l1_conv = self._conv_layer(X_image, self.network_weights['W_conv1'],
                                       self.network_weights['b_conv1'])
            l1_pool = self._max_pool_2x2(l1_conv, name='h_conv1')
        with tf.name_scope('Conv2'):
            l2_conv = self._conv_layer(l1_pool, self.network_weights['W_conv2'],
                                       self.network_weights['b_conv2'])
            l2_pool = self._max_pool_2x2(l2_conv, name='h_conv2')
        l2_pool_flat = tf.reshape(l2_pool,[-1, self.network_architecture['W_fc1'][0]])
        with tf.name_scope('Dense'):
            l1_fc = tf.nn.relu(tf.matmul(l2_pool_flat, self.network_weights['W_fc1']) +
                               self.network_weights['b_fc1'],
                               name='h_fc1')
            l1_fc_dropout = tf.nn.dropout(l1_fc, self.keep_prob)
            self.y_conv = tf.matmul(l1_fc_dropout, self.network_weights['W_fc2']) + \
                          self.network_weights['b_fc2']
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('AccuracyScore', self.accuracy)
        return self.y_conv

    def _create_loss_optimizer(self):
        with tf.name_scope('Cross_Entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y_))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        return self.train_step

    def _initialize_weights(self, W_conv1, W_conv2, W_fc1, W_fc2):
        # all_weights = {
        #     'W_conv1': tf.get_variable('W_conv1', shape=W_conv1,
        #                                initializer=tf.contrib.layers.xavier_initializer()),
        #     'W_conv2': tf.get_variable('W_conv2', shape=W_conv1,
        #                                initializer=tf.contrib.layers.xavier_initializer()),
        #     'W_fc1': tf.get_variable('W_fc1', shape=W_conv1,
        #                                initializer=tf.contrib.layers.xavier_initializer()),
        #     'W_fc2': tf.get_variable('W_fc2', shape=W_conv1,
        #                                initializer=tf.contrib.layers.xavier_initializer()),
        #     'b_conv1': tf.get_variable('b_conv1', initializer=tf.constant(0.1, shape=[W_conv1[3]])),
        #     'b_conv2': tf.get_variable('b_conv2', initializer=tf.constant(0.1, shape=[W_conv2[3]])),
        #     'b_fc1': tf.get_variable('b_fc1', initializer=tf.constant(0.1, shape=[W_fc1[1]])),
        #     'b_fc2': tf.get_variable('b_fc2', initializer=tf.constant(0.1, shape=[W_fc2[1]])),
        # }
        with tf.name_scope('Weights_N_Biases'):
            all_weights = {
                'W_conv1': tf.Variable(tf.truncated_normal(W_conv1, stddev=0.1), name='W_conv1'),
                'W_conv2': tf.Variable(tf.truncated_normal(W_conv2, stddev=0.1), name='W_conv2'),
                'W_fc1': tf.Variable(tf.truncated_normal(W_fc1, stddev=0.1), name='W_fc1'),
                'W_fc2': tf.Variable(tf.truncated_normal(W_fc2, stddev=0.1), name='W_fc2'),
                'b_conv1': tf.Variable(tf.constant(0.1, shape=[W_conv1[3]]),name='b_conv1'),
                'b_conv2': tf.Variable(tf.constant(0.1, shape=[W_conv2[3]]),name='b_conv2'),
                'b_fc1': tf.Variable(tf.constant(0.1, shape=[W_fc1[1]]),name='b_fc1'),
                'b_fc2': tf.Variable(tf.constant(0.1, shape=[W_fc2[1]]),name='b_fc2')
            }
        return all_weights

    def _conv_layer(self, input, weights, biases, strides=[1, 1, 1, 1] , padding='SAME', name=None):
        conv2d = tf.nn.conv2d(input, weights, strides=strides, padding=padding)
        h_conv = tf.nn.relu(conv2d + biases, name=name)
        return h_conv

    def _max_pool_2x2(self, input, name=None):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _initialize_histograms(self, hist_list):
        self.all_hists = dict()
        for hist in hist_list:
            self.all_hists[hist] = tf.summary.histogram(hist, self.network_weights[hist])
        return self.all_hists

    def color_distorer(self, image, thread_id=0, scope=None):
        #TODO: Decide if this should go in TFCoder or TFHelper
        with tf.name_scope('distort_color'):
            color_ordering = thread_id % 2

            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)

                # The random_* ops do not necessarily clamp.
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    # def get_accuracy(self):
    #     correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
    #     self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     return self.accuracy

    def partial_fit(self, X, y):
        opt, cost = self.sess.run(self.train_step, feed_dict={self.x: X, self.y_: y, self.keep_prob: 1.0})
        return cost

    def predict(self, X):
        return self.sess.run(self.y_conv, feed_dict={self.x: X, self.keep_prob: 1.0})

    def train(self, training_data_dict, epochs=500, TFRecords=True):
        # TODO: make it show prediction labels on test images
        self.training_data_dict = training_data_dict
        distorted_image = tf.reshape(training_data_dict['X_train_batch'], [-1, 160, 160, 3]) #1
        distorted_image = tf.map_fn(lambda img: self.color_distorer(img), distorted_image) #2
        distorted_image = tf.reshape(distorted_image, [-1, 76800])  # 3
        for epoch in range(epochs):
            if TFRecords is True:
                x_test, y_test = self.sess.run([training_data_dict['X_test_batch'], training_data_dict['y_test_batch']])
                distortedX, x_train, y_train = self.sess.run([distorted_image, training_data_dict['X_train_batch'],
                                                              training_data_dict['y_train_batch']]) #4
            else:
                x_test, y_test = training_data_dict['X_test_batch'], training_data_dict['y_test_batch']
                x_train, y_train = training_data_dict['X_train_batch'], training_data_dict['y_train_batch']
                # TODO: make a batch iterator for non-TFRecords
            summary, loss, train_accuracy = self.sess.run([self.merged, self.train_step, self.accuracy], feed_dict={
                self.x: x_train, self.y_: y_train, self.keep_prob: self.dropout})
            self.train_writer.add_summary(summary, epoch)
            if epoch%300 == 0:
                print('Distorting color in training images...')
                distortedX, x_train, y_train = self.sess.run([distorted_image, training_data_dict['X_train_batch'],
                                                              training_data_dict['y_train_batch']])
                summary, loss, train_accuracy = self.sess.run([self.merged, self.train_step, self.accuracy], feed_dict={
                    self.x: distortedX, self.y_: y_train, self.keep_prob: 5.0})
                print("Step %d, Distorted Training accuracy %g" % (epoch, train_accuracy))
            if epoch%100 == 0:
                summary, test_accuracy = self.sess.run([self.merged, self.accuracy], feed_dict={
                    self.x: x_test, self.y_: y_test, self.keep_prob: 1.0})
                print("Step %d, Training accuracy %g" % (epoch, train_accuracy))
                print("Test Accuracy %g"% test_accuracy)
                self.test_writer.add_summary(summary, epoch)
            if epoch%500 == 0:
                print('Saving model...')
                self.saver.save(self.sess, self.model_dir+'model.ckpt', epoch)
            if epoch == epochs:
                print('Saving model...')
                self.saver.save(self.sess, self.model_dir+'model.ckpt', epoch)








