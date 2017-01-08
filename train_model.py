from ConvNetClass import BasicConvNet
import tensorflow as tf
from TF_Helper import inputs


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('n_classes', 2, 'Number of classes in the data.')
flags.DEFINE_integer('batch_size', 30, 'Mini-Batch Size.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for training dropout.')
flags.DEFINE_string('train_file', 'garage_door160_TRAIN.tfrecords', 'Name of the TFRecords file used for training.')
flags.DEFINE_string('test_file', 'garage_door160_TEST.tfrecords', 'Name of the TFRecords file used for testing.')
flags.DEFINE_string('train_dir','/media/mcamp/Local SSHD/Python Projects/GarageDoor2',
                    'Path to the directory housing training data and other project files.')
flags.DEFINE_integer('n_epochs', 50000000, 'The number of epochs the model is to run.')

training_data = dict()
training_data['X_train_batch'], training_data['y_train_batch'] = inputs(FLAGS.train_dir,
                                                                  FLAGS.train_file,
                                                                  FLAGS.batch_size,
                                                                  FLAGS.n_epochs,
                                                                  FLAGS.n_classes,
                                                                  one_hot_labels=True,
                                                                  imshape=76800)


training_data['X_test_batch'], training_data['y_test_batch'] = inputs(FLAGS.train_dir,
                                                                FLAGS.test_file,
                                                                FLAGS.batch_size,
                                                                FLAGS.n_epochs,
                                                                FLAGS.n_classes,
                                                                one_hot_labels=True,
                                                                imshape=76800)

network_architecture = dict(W_conv1=[5, 5, 3, 25],
                            W_conv2=[5, 5, 25, 50],
                            W_fc1=[40 * 40 * 50, 1024],
                            W_fc2=[1024, 2])

model = BasicConvNet(76800, 2, network_architecture, batch_size=30,
                     imshape=(160,160,3), dropout=0.5, model_dir='./modeltest/')
model.train(training_data, epochs=FLAGS.n_epochs)
#
