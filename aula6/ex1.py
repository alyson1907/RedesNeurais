# Name: Alyson Matheus Maruyama Nascimento -8532269

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# NOTE: Even though the problem asks for a single avaliation method, 
# I am implementing both Holdout and Cross-Validation methods for `mlp1` and `mlp2`.
# `mlp1` and `mlp2` use the same MLP Architecture (Architecture 1), 
# whilist `mlp3` will use a different Architecture, used to compare both models

# Reading CSV and settings first row as header (column names)
dados1 = pd.read_csv('./dados2.csv', header=0)

X = np.array(dados1.drop(['V3'], axis=1))
y = np.array(dados1['V3'])

# Holdout method: simply splits the dataset into Train/Test, 
# using the Train subset to train the model and the test susbset to validate it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
mlp1.fit(X_train, y_train)
mlp1_predicted = mlp1.predict(X_test)

svm1 = svm.SVC(kernel='linear')
svm1.fit(X_train, y_train)
svm1_predicted = svm1.predict(X_test)

# Dados1
print('=================================================')
print('=====================  MLP  =====================')
print('=================================================')

print('-------------------  Dados1.csv  ----------------')
print('\n- MLP Accuracy:', metrics.accuracy_score(y_test, mlp1_predicted))
print('\n- MLP Confusion Matrix:\n', metrics.confusion_matrix(y_test, mlp1_predicted))
print('\n- MLP Classification Report\n', metrics.classification_report(y_test, mlp1_predicted))


print('=================================================')
print('=====================  SVM  =====================')
print('=================================================')

print('-------------------  Dados1.csv  ----------------')
print('\n- SVM Accuracy:', metrics.accuracy_score(y_test, svm1_predicted))
print('\n- SVM Confusion Matrix:\n', metrics.confusion_matrix(y_test, svm1_predicted))
print('\n- SVM Classification Report\n', metrics.classification_report(y_test, svm1_predicted))
