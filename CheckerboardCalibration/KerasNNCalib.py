from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np

datapath = 'Dataset\\CheckboarderData_perspective.csv'
raw_data = np.loadtxt(datapath, delimiter = ',')

input_data = np.ones((raw_data.shape[0], 3), dtype=np.float32)
output_label = np.ones((raw_data.shape[0], 3), dtype=np.float32)

input_data[:, 0:2] = raw_data[:, 2:4].astype(np.float32)
output_label[:, 0:2] = raw_data[:, 0:2].astype(np.float32)

order = np.argsort(np.random.random((input_data.shape[0],)))
input_data = input_data[order]
output_label = output_label[order]

train_num = 300
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

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.sigmoid,
                       input_shape=(train_data.shape[1],)),
    #keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(3)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.03)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mse'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_label, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error pixels')
  plt.plot(history.epoch, np.array(history.history['mean_squared_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_squared_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 100])

plot_history(history)
plt.show()