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

print(parseColumns(loadCsv('./Aula3-dataset_1.csv')))
		
