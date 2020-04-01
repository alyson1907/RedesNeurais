# Name: Alyson Matheus Maruyama Nascimento -8532269

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# NOTE: Even though the problem asks for a single avaliation method, 
# I am implementing both Holdout and Cross-Validation methods for `mlp1` and `mlp2`.
# `mlp1` and `mlp2` use the same MLP Architecture (Architecture 1), 
# whilist `mlp3` will use a different Architecture, used to compare both models

# Reading CSV and settings first row as header (column names)
dataset = pd.read_csv('./Dataset_3Cluster_4features.csv', header=0)

X = np.array(dataset.drop(['V5'], axis=1))
y = np.array(dataset['V5'])

# Holdout method: simply splits the dataset into Train/Test, 
# using the Train subset to train the model and the test susbset to validate it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
mlp1.fit(X_train, y_train)
print('\nMLP Accuracy after validating (R_est):', mlp1.score(X_test, y_test))
# ----------------------------------------------------------------------------------
# Cross Validatiion method
mlp2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
scores = cross_val_score(mlp2, X, y, cv=10) # cv is the number of groups
print('\n\nScores for each iteraction:', scores)

# The final accuracy for Cross Validation is the average of the score values computed
print("\n Final Accuracy(R_est) for Cross Validation: %0.2f" % scores.mean())


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# Implementing Holdout method for Architecture 2
mlp3 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(7, 3), learning_rate_init=0.05, shuffle=True, random_state=50)
mlp3.fit(X_train, y_train)
print('\nArchitecture 2 after validating (R_est):', mlp3.score(X_test, y_test))

# When Running the program multiple times, usually Architecture 2 (mlp3) is 
# better than Architecture 1 (`mlp1`) when using Holdout method for avaliation
