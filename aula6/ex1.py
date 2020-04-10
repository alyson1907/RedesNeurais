# Name: Alyson Matheus Maruyama Nascimento -8532269

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Reading CSV and settings first row as header (column names)
dados1 = pd.read_csv('./dados1.csv', header=0)
dados2 = pd.read_csv('./dados2.csv', header=0)

fig, axs = plt.subplots(2, figsize=(6,9))
# Plotting the initial data distribution
axs[0].set_title('Distribuição de dados1')
axs[1].set_title('Distribuição de dados2')
axs[0].plot([dados1['V1']], [dados1['V2']], marker='o', markersize=1, color="red")
axs[1].plot([dados2['V1']], [dados2['V2']], marker='o', markersize=1, color="blue")
plt.show()

X1 = np.array(dados1.drop(['V3'], axis=1))
y1 = np.array(dados1['V3'])

X2 = np.array(dados2.drop(['V3'], axis=1))
y2 = np.array(dados2['V3'])

# Holdout method: simply splits the dataset into Train/Test, 
# using the Train subset to train the model and the test susbset to validate it
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.7)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.7)

mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
mlp1.fit(X1_train, y1_train)
mlp1_predicted = mlp1.predict(X1_test)

svm1 = svm.SVC(kernel='linear')
svm1.fit(X1_train, y1_train)
svm1_predicted = svm1.predict(X1_test)

# Using the same architectures from above but to classify `dados2.csv`
mlp2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
mlp2.fit(X2_train, y2_train)
mlp2_predicted = mlp2.predict(X2_test)

svm2 = svm.SVC(kernel='linear')
svm2.fit(X2_train, y2_train)
svm2_predicted = svm2.predict(X2_test)

# Dados1
print('=================================================')
print('=====================  MLP  =====================')
print('=================================================')
print('-------------------  Dados1.csv  ----------------')
print('\n- MLP1 Accuracy:', metrics.accuracy_score(y1_test, mlp1_predicted))
print('\n- MLP1 Confusion Matrix:\n', metrics.confusion_matrix(y1_test, mlp1_predicted))
print('\n- MLP1 Classification Report\n', metrics.classification_report(y1_test, mlp1_predicted))

print('-------------------  Dados2.csv  ----------------')
print('\n- MLP2 Accuracy:', metrics.accuracy_score(y2_test, mlp2_predicted))
print('\n- MLP2 Confusion Matrix:\n', metrics.confusion_matrix(y2_test, mlp2_predicted))
print('\n- MLP2 Classification Report\n', metrics.classification_report(y2_test, mlp2_predicted))


print('=================================================')
print('=====================  SVM  =====================')
print('=================================================')
print('-------------------  Dados1.csv  ----------------')
print('\n- SVM1 Accuracy:', metrics.accuracy_score(y1_test, svm1_predicted))
print('\n- SVM1 Confusion Matrix:\n', metrics.confusion_matrix(y1_test, svm1_predicted))
print('\n- SVM1 Classification Report\n', metrics.classification_report(y1_test, svm1_predicted))

print('-------------------  Dados2.csv  ----------------')
print('\n- SVM2 Accuracy:', metrics.accuracy_score(y2_test, svm2_predicted))
print('\n- SVM2 Confusion Matrix:\n', metrics.confusion_matrix(y2_test, svm2_predicted))
print('\n- SVM2 Classification Report\n', metrics.classification_report(y2_test, svm2_predicted))
