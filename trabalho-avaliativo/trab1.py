# Name: Alyson Matheus Maruyama Nascimento -8532269

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
from sklearn import svm

# Reading IRIS dataset from sklearn
dataset = datasets.load_iris()

X = dataset.data[:, :-1]
y = dataset.target

# Holdout method: simply splits the dataset into Train/Test, 
# using the Train subset to train the model and the test susbset to validate it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# First classifier: MLP
mlp1 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 3, 3), activation='logistic', learning_rate='constant', learning_rate_init=0.002)
mlp1.fit(X_train, y_train)
mlp1_predicted = mlp1.predict(X_test)
print('\n- MLP1 Accuracy:', metrics.accuracy_score(y_test, mlp1_predicted))
print('\n- MLP1 Confusion Matrix:\n', metrics.confusion_matrix(y_test, mlp1_predicted))
print('\n- MLP1 Classification Report\n', metrics.classification_report(y_test, mlp1_predicted))

# Second classifier using the same technique (Holdout)
svm1 = svm.SVC(kernel='linear')
svm1.fit(X_train, y_train)
svm1_predicted = svm1.predict(X_test)
print('\n- SVM1 Accuracy:', metrics.accuracy_score(y_test, svm1_predicted))
print('\n- SVM1 Confusion Matrix:\n', metrics.confusion_matrix(y_test, svm1_predicted))
print('\n- SVM1 Classification Report\n', metrics.classification_report(y_test, svm1_predicted))
