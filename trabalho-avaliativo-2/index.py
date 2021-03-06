# Name: Alyson Matheus Maruyama Nascimento - 8532269

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loads mnist data, concatenates into a single dataset and splits it
def split_train_test(train_size = 0.1):
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
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

  # Normalizing input
  x_train /= 255
  x_test /= 255

  return x_train, x_test, y_train, y_test

def create_CNN(input_shape):
  # Creating a Sequential Model
  cnn = Sequential()
  # Adding Layers to the CNN
  cnn.add(Conv2D(28, kernel_size=(8,4), input_shape=input_shape))
  cnn.add(MaxPooling2D(pool_size=(2, 2)))
  cnn.add(Flatten())
  cnn.add(Dense(128, activation=tf.nn.relu))
  cnn.add(Dropout(0.2))
  cnn.add(Dense(10,activation=tf.nn.softmax))

  cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return cnn

def get_metrics_from_training(hist):
  # Array: each position is related to a epoch
  epoch_accuracies = hist.history.get('accuracy') 
  epoch_losses = hist.history.get('loss') 
  return epoch_accuracies, epoch_losses

################ Main Program ################
input_shape = (28, 28, 1)
# Parameters
train_percentage = 0.4
epochs = 15

x_train, x_test, y_train, y_test = split_train_test(train_percentage)
x_train, x_test, y_train, y_test = convert_and_format_data(x_train, x_test, y_train, y_test)
cnn = create_CNN(input_shape)

# Training the model for x epochs
hist = cnn.fit(x=x_train,y=y_train, epochs=epochs)
# `accs` will contain accuracy metrics from training step
training_accs, training_losses = get_metrics_from_training(hist)

# `test_loss, test_acc` will contain accuracy from test step
test_loss, test_acc = cnn.evaluate(x_test, y_test)

print('Training Accuracy for each epoch', training_accs)
print('Training Loss for each epoch', training_losses)
print('Final test accuracy', test_acc)

epoch_count = range(1, epochs + 1)
# `ax` is an array containing each row,
# and each `ax[row]` contains the columns, which are a `plt.` instance
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,8))
subplot1 = ax[0]
subplot2 = ax[1]
# Setting only integer values for `epoch` axis
subplot1.set_xticks(epoch_count)
subplot2.set_xticks(epoch_count)
subplot1.set_title('CNN Training Accuracy')
subplot2.set_title('CNN Training Loss')
# Setting axes labels
subplot1.set_xlabel('epoch')
subplot2.set_xlabel('epoch')
subplot1.set_ylabel('accuracy')
subplot2.set_ylabel('loss')

subplot1.plot(epoch_count, training_accs, label='Training Accuracy')
subplot2.plot(epoch_count, training_losses, label='Training Losses')
# plt.plot(test_acc, label='Test Accuracy')
subplot1.legend()
subplot2.legend()
plt.show()
