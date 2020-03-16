from csv import reader
import random
import numpy as np 
from matplotlib import pyplot as plt

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

# Threshold activation function
def activationFuncThreshold(row, weights):
  # bias
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 2.0 if activation >= 0.0 else 1.0

def activationFuncLinear(row, weights):
  # bias
  activation = weights[0]
  for i in range(len(row)-1):
    activation += weights[i + 1] * row[i]
  if activation <= 0.0:
    return 1.0
  elif activation > 0.0 and activation < 1.0:
    return activation
  else:
    return 2.0

# Estimate Perceptron weights
def trainWeightsPerceptron(trainingDataset, learningRate, epochs):
  # Initializing weights with random value
  weights = [random.randint(0, 30) for i in range(len(trainingDataset[0]))]
  errors = []
  for epoch in range(epochs):
    currentError = 0
    for row in trainingDataset:
      prediction = activationFuncThreshold(row, weights)
      error = prediction - row[-1]
      currentError += error**2
      errors.append(currentError)
      # Updating new bias
      weights[0] = weights[0] - (learningRate * error)
      for i in range(len(row)-1):
        weights[i + 1] = weights[i + 1] - (learningRate * row[i] * error)
    print('Epoch=%d, learningRate=%.4f, currentError=%.4f' % (epoch, learningRate, currentError))
    print('weights =', weights)
    print('\n')
  return [weights, errors]

def trainWeightsAdaline(trainingDataset, learningRate, epochs):
  # Initializing weights with random value
  weights = [random.randint(0, 30) for i in range(len(trainingDataset[0]))]
  errors = []
  for epoch in range(epochs):
    currentError = 0
    for row in trainingDataset:
      prediction = activationFuncLinear(row, weights)
      error = row[-1] - prediction
      currentError += error**2
      errors.append(currentError)
      # Updating new bias
      weights[0] = weights[0] + (learningRate * error)
      # Update the weights based on the delta rule
      for i in range(len(row)-1):
        weights[i + 1] = weights[i + 1] + (learningRate * row[i] * error)
    print('Epoch=%d, learningRate=%.4f, currentError=%.4f' % (epoch, learningRate, currentError))
    print('weights =', weights)
    print('\n')
  return [weights, errors]

def Perceptron(trainDataset, testDataset, learningRate = 0.04, epochs = 70):
  predictions = list()
  weights, errors = trainWeightsPerceptron(trainDataset, learningRate, epochs)

  for row in testDataset:
    prediction = activationFuncThreshold(row, weights)
    predictions.append(prediction)
  return [predictions, errors]

def Adaline(trainDataset, testDataset, learningRate = 0.04, epochs = 80):
  predictions = list()
  weights, errors = trainWeightsAdaline(trainDataset, learningRate, epochs)

  for row in testDataset:
    prediction = activationFuncLinear(row, weights)
    predictions.append(prediction)
  return [predictions, errors]

###################################################
#                   MAIN PROGRAM                   #
###################################################
fullDataset = parseColumns(loadCsv('./Aula3-dataset_1.csv'))
# splitting into train/test sets
trainDataset = []
for i in range(0, len(fullDataset), 3):
  trainDataset.append(fullDataset[i])

perceptronPredictions, errors = Perceptron(trainDataset, fullDataset)
adalinePredictions, errors = Adaline(trainDataset, fullDataset)

# Calculating accuracy
expectedArr = [row[-1] for row in fullDataset]
print('Perceptron Accuracy: %.4f' % accuracy(expectedArr, perceptronPredictions))
print('Adaline Accuracy: %.4f' % accuracy(expectedArr, adalinePredictions))
