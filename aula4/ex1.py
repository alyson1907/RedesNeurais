# Nome: Alyson Matheus Maruyama Nascimento - 8532269
from csv import reader
import tensorflow as tf

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

dataset = parseColumns(loadCsv('./Dataset_3Cluster_4features.csv'))

# Parameters
learningRate = 0.001
epochs = 15
batchSize = 10
displayStep = 1

numberNeuronsLayer1 = 5
numberNeuronsLayer2 = 5
inputSize = len(dataset) # excluding first row with column names
outputSize = inputSize

x = tf.compat.v1.placeholder(tf.float32, shape = [None, inputSize])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, outputSize])

activationFunc = tf.nn.tanh

# Weights
weights = {
    'layer1': tf.Variable(tf.random.normal([inputSize, numberNeuronsLayer1])),
    'layer2': tf.Variable(tf.random.normal([numberNeuronsLayer1, numberNeuronsLayer2])),
    'output': tf.Variable(tf.random.normal([numberNeuronsLayer2, outputSize]))
}

# Biases
biases = {
    'b1': tf.Variable(tf.random.normal([numberNeuronsLayer1])),
    'b2': tf.Variable(tf.random.normal([numberNeuronsLayer2])),
    'output': tf.Variable(tf.random.normal([outputSize]))
}

def MLP(row):
  outputLayer1 = tf.add(tf.matmul(row, weights['layer1']), biases['b1'])
  outputLayer2 = tf.add(tf.matmul(outputLayer1, weights['layer2']), biases['b2'])
  output = tf.add(tf.matmul(outputLayer2, weights['output']), biases['output'])

  print('\n\n\n\n\n', output)
  return

# Constructing the model
asd = MLP(x)
tf.print(asd)
