# Name: Alyson Matheus Maruyama Nascimento -8532269

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import fetch_openml

# Reading IRIS dataset from sklearn
print('Fetching MNIST...')
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
print('Done')

# Holdout method: simply splits the dataset into Train/Test, 
# using the Train subset to train the model and the test susbset to validate it
print('Splitting MNIST dataset...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)
print('Done')

# First classifier: MLP
mlp1 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 3), activation='logistic', learning_rate='constant', learning_rate_init=0.002)
print('Training MLP1...')
mlp1.fit(X_train, y_train)
print('Predicting MLP1...')
mlp1_predicted = mlp1.predict(X_test)

print('Calculating scores MLP1...')
print('\n- MLP1 Accuracy:', metrics.accuracy_score(y_test, mlp1_predicted))
print('\n- MLP1 Confusion Matrix:\n', metrics.confusion_matrix(y_test, mlp1_predicted))
print('\n- MLP1 Classification Report\n', metrics.classification_report(y_test, mlp1_predicted))

# Second classifier using the same technique (Holdout)
mlp2 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 20, 20, 10), activation='relu', learning_rate='constant', learning_rate_init=0.004)
print('Training MLP2...')
mlp2.fit(X_train, y_train)
print('Predicting MLP2...')
mlp2_predicted = mlp2.predict(X_test)

print('Calculating scores MLP2...')
print('\n- MLP2 Accuracy:', metrics.accuracy_score(y_test, mlp2_predicted))
print('\n- MLP2 Confusion Matrix:\n', metrics.confusion_matrix(y_test, mlp2_predicted))
print('\n- MLP2 Classification Report\n', metrics.classification_report(y_test, mlp2_predicted))
