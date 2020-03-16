from csv import reader
 
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
def predict(row, weights):
  # bias
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 2.0 if activation >= 0.0 else 1.0

# Estimate Perceptron weights
def trainWeights(trainingDataset, learningRate, epochs):
  weights = [0.0 for i in range(len(trainingDataset[0]))]
  for epoch in range(epochs):
    for row in trainingDataset:
      prediction = predict(row, weights)
      error = prediction - row[-1]
      # Updating new bias
      weights[0] = weights[0] - (learningRate * error)
      for i in range(len(row)-1):
        weights[i + 1] = weights[i + 1] - (learningRate * row[i] * error)
    print('Epoch=%d, learningRate=%.4f' % (epoch, learningRate))
    print('weights =', weights)
  return weights

def Perceptron(trainDataset, testDataset, learningRate = 0.04, epochs = 3):
  predictions = list()
  weights = trainWeights(trainDataset, learningRate, epochs)

  for row in testDataset:
    prediction = predict(row, weights)
    predictions.append(prediction)
  return predictions

###################################################
#                   MAIN PROGRAM                   #
###################################################
fullDataset = parseColumns(loadCsv('./Aula3-dataset_1.csv'))
# splitting into train/test sets
trainDataset = []
for i in range(0, len(fullDataset), 3):
  trainDataset.append(fullDataset[i])

perceptronPredictions = Perceptron(trainDataset, fullDataset)

# Calculating accuracy
expectedArr = [row[-1] for row in fullDataset]
print('Perceptron Accuracy: %.4f' % accuracy(expectedArr, perceptronPredictions))
