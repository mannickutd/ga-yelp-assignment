import os
import numpy as np
import tensorflow as tf
from sklearn import cross_validation, metrics
from sklearn.datasets.base import Bunch
from tensorflow.contrib import skflow
from flask import current_app


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')


def conv_model(X, y):
    x_dim = current_app.config['IMAGE_X_DIM']
    y_dim = current_app.config['IMAGE_Y_DIM']
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and height
    # final dimension being the number of color channels
    X = tf.reshape(X, [-1, x_dim, y_dim, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], 
                                    bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5], 
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, int((x_dim/4)) * int((y_dim/4)) * 64])
    # densely connected layer with 1024 neurons
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    return skflow.models.logistic_regression(h_fc1, y)


###
# Learning rate options
# from 0.05 to 0.025 slows performance considerably, increased ~5% in accuracy.

# Steps options
# from 1000 to 500 causes ~5% drop in accuracy, half the time to compute.

# Batch size
# from 100 to 1000 consumed too much memory and started using the swap file.
###
def get_classifier():
    run_config = skflow.estimators.RunConfig(
        num_cores=current_app.config.get('NUM_CORES'),
        gpu_memory_fraction=current_app.config.get('GPU_MEMORY_FRACTION'))
    return skflow.TensorFlowEstimator(
        model_fn=conv_model,
        n_classes=current_app.config.get('NUM_CLASSES'),
        steps=current_app.config.get('STEPS'),
        learning_rate=current_app.config.get('LEARNING_RATE'),
        config=run_config
    )


## General usage is classifier.get_tensor_value('foo')
## 'foo' must be the variable scope of the desired tensor followed by the
## graph path. 

## To understand the mechanism and figure out the right scope and path, you can do logging.
## Then use TensorBoard or a text editor on the log file to look at available strings.

## First Convolutional Layer
#print('1st Convolutional Layer weights and Bias')
#print(classifier.get_tensor_value('conv_layer1/convolution/filters:0'))
#print(classifier.get_tensor_value('conv_layer1/convolution/bias:0'))

## Second Convolutional Layer
#print('2nd Convolutional Layer weights and Bias')
#print(classifier.get_tensor_value('conv_layer2/convolution/filters:0'))
#print(classifier.get_tensor_value('conv_layer2/convolution/bias:0'))

## Densely Connected Layer
#print('Densely Connected Layer weights')
#print(classifier.get_tensor_value('dnn/layer0/Linear/Matrix:0'))

## Logistic Regression weights
#print('Logistic Regression weights')
#print(classifier.get_tensor_value('logistic_regression/weights:0'))
