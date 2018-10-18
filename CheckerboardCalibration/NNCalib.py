from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

datapath = 'Dataset\\CheckboardData_perspective.csv'
raw_data = np.loadtxt(datapath, delimiter = ',')

input_data = np.ones((raw_data.shape[0], 3), dtype=np.float32)
output_label = np.ones((raw_data.shape[0], 3), dtype=np.float32)

input_data[:, 0:2] = raw_data[:, 2:4].astype(np.float32)
output_label[:, 0:2] = raw_data[:, 0:2].astype(np.float32)

order = np.argsort(np.random.random((input_data.shape[0],)))
input_data = input_data[order]
output_label = output_label[order]

train_num = 250
valid_num = 50
test_num = 50

train_data = input_data[0:train_num, :]
valid_data = input_data[train_num : train_num + valid_num, :]
test_data = input_data[train_num + valid_num : -1, :]

train_label = output_label[0:train_num, :]
valid_label = output_label[train_num : train_num + valid_num, :]
test_label = output_label[train_num + valid_num : -1, :]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

mean[2] = 0
std[2] = 1

train_data = (train_data - mean) / std
valid_data = (valid_data - mean) / std
test_data = (test_data - mean) / std

param_length = 3
num_neuron = 256

graph = tf.Graph()
with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(train_data)
    tf_train_labels = tf.constant(train_label)
    tf_valid_dataset = tf.constant(valid_data)
    tf_valid_labels = tf.constant(valid_label)
    tf_test_dataset = tf.constant(test_data)
    tf_test_labels = tf.constant(test_label)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights_1 = tf.Variable(
        tf.truncated_normal([param_length, num_neuron]))
    biases_1 = tf.Variable(tf.zeros([num_neuron]))

    weights_2 = tf.Variable(
        tf.truncated_normal([num_neuron, param_length]))
    biases_2 = tf.Variable(tf.zeros([param_length]))

    def get_model(input_set):
        activation = tf.nn.sigmoid(tf.matmul(input_set, weights_1) + biases_1)
        prediction = tf.matmul(activation, weights_2) + biases_2
        return prediction

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    train_prediction = get_model(tf_train_dataset)
    loss = tf.losses.mean_squared_error(labels=tf_train_labels, predictions=train_prediction)

    print (tf.trainable_variables())

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    valid_prediction = get_model(tf_valid_dataset)
    test_prediction = get_model(tf_test_dataset)
    valid_MSE = tf.losses.mean_squared_error(labels = tf_valid_labels, predictions = valid_prediction)
    test_MSE = tf.losses.mean_squared_error(labels = tf_test_labels, predictions = test_prediction)

num_steps = 500
epoch = 200
with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  best_train_MSE = float('Inf')
  best_valid_MSE = float('Inf')
  best_test_MSE = float('Inf')
  for iteration in range(epoch):
      tf.global_variables_initializer().run()
      #print('Initialized')
      for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if l < best_train_MSE:
            best_train_MSE = l
      if valid_MSE.eval() < best_valid_MSE:
          best_valid_MSE = valid_MSE.eval()
      if (iteration % 20 == 0):
          print('Best Train MSE so far: %.1f' % best_train_MSE)
          print('Validation MSE so far: %.1f' % valid_MSE.eval())

  print('Best Train MSE: %.1f' % best_train_MSE)
  print('Best Validation MSE: %.1f' % best_valid_MSE)
      #test_pred = test_prediction.eval()
  a = 1