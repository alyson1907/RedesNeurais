# Nome: Alyson Matheus Maruyama Nascimento - 8532269
from csv import reader
# Class MLPClassifier implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation
# MLPClassifier documentation https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import random
import numpy as np

# Load a CSV file
def loadCsv(filename):
  dataset = list()
  with open(filename, 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
      if not row:
        continue
      dataset.append(row)
  dataset.pop(0)
  return dataset

# Convert string column to float
def parseColumns(dataset):
  for row in dataset:
    for i in range(len(row)):
      row[i] = float(row[i].strip())
  return dataset

# Accuracy calculation
def accuracy(expected, predicted):
	correct = 0
	for i in range(len(expected)):
		if expected[i] == predicted[i]:
			correct += 1
	return correct / float(len(expected)) * 100.0

dataset = parseColumns(loadCsv('./Dataset_3Cluster_4features.csv'))

# X = training dataset
X = []
i = 0
while i <= len(dataset) - 3:
  X.append(dataset.pop(i))
  i+= 3

# y = expected output for each row
y = [row.pop(-1) for row in X]
print('Training size:', len(X))
print('Dataset size:', len(dataset))

# Setting up MLPs with different inputs
mlp1 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(15, 4), random_state=1,
learning_rate_init=0.002,
learning_rate='constant',
shuffle=True)

mlp2 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(30, 3), random_state=1,
learning_rate_init=0.001,
learning_rate='constant',
shuffle=True)

mlp3 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(50, 10), random_state=0,
learning_rate_init=0.012,
learning_rate='adaptive',
shuffle=True)

# Training the MLPs
mlp1.fit(X, y)
mlp2.fit(X, y)
mlp3.fit(X, y)

expected = [row.pop(-1) for row in dataset]
mlp1Predictions = mlp1.predict([row for row in dataset])
mlp2Predictions = mlp2.predict([row for row in dataset])
mlp3Predictions = mlp3.predict([row for row in dataset])

print('Accuracy for MLP1:', accuracy(expected, mlp1Predictions))
print('Accuracy for MLP2:', accuracy(expected, mlp2Predictions))
print('Accuracy for MLP3:', accuracy(expected, mlp3Predictions))

plt.title('Loss Curves (Learning)')
plt.ylabel("Loss")
plt.plot(mlp1.loss_curve_)
plt.plot(mlp2.loss_curve_)
plt.plot(mlp3.loss_curve_)
plt.show()
