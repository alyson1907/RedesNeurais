# Name: Alyson Matheus Maruyama Nascimento - 8532269

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loads mnist data, concatenates into a single dataset and splits it
def split_train_test(train_size = 0.1):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  # Concatenating to create whole MNIST dataset (will be spliited later)
  x = np.concatenate((x_train, x_test))
  y = np.concatenate((y_train, y_test))
  x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
  return x_train, x_test, y_train, y_test

def convert_and_format_data(x_train, x_test, y_train, y_test):
  # Reshaping the arrays: it is needed to have 4 dimentions
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

  # Converting data type from uint8 --> float in order to compute divisions with decimal numbers
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  x_train /= 255
  x_test /= 255

  return x_train, x_test, y_train, y_test

def create_CNN(input_shape):
  # Creating a Sequential Model
  cnn = Sequential()
  # Adding Layers to the CNN
  cnn.add(Conv2D(28, kernel_size=(4,4), input_shape=input_shape))
  cnn.add(MaxPooling2D(pool_size=(2, 2)))
  cnn.add(Flatten()) # Flattening the 2D arrays for fully connected layers
  cnn.add(Dense(128, activation=tf.nn.relu))
  cnn.add(Dropout(0.2))
  cnn.add(Dense(10,activation=tf.nn.softmax))

  cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return cnn

def get_metrics_from_training(hist):
  # Array: each position is related to a epoch
  epoch_accuracies = hist.history.get('accuracy') 
  return epoch_accuracies

################ Main Program ################
input_shape = (28, 28, 1)
train_sizes = []
output_training_metrics = []
output_test_metrics = []

for i in range(1, 10):
  x_train, x_test, y_train, y_test = split_train_test(i / 10)
  x_train, x_test, y_train, y_test = convert_and_format_data(x_train, x_test, y_train, y_test)
  cnn = create_CNN(input_shape)

  # Training the model for only 2 epochs
  hist = cnn.fit(x=x_train,y=y_train, epochs=1)
  # `accs` will contain accuracy metrics from training step
  accs = get_metrics_from_training(hist)

  # `test_loss, test_acc` will contain accuracy from test step
  test_loss, test_acc = cnn.evaluate(x_test, y_test)
  train_sizes.append(str(i * 10) + '%')
  output_training_metrics.append(accs[-1])
  output_test_metrics.append(test_acc)

print('output_training_metrics', output_training_metrics)
print('output_test_metrics', output_test_metrics)

plt.title('Training and Test Accuracy - CNN')
plt.plot(train_sizes, output_training_metrics, label='Training Accuracy')
plt.plot(train_sizes, output_test_metrics, label='Test Accuracy')
plt.legend()
plt.show()
