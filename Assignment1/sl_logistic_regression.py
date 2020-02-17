
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Importing TRAINING data
data = np.load("fashion_train.npy")
x_train = np.array([x[:-1] for x in data])/255
y_train = np.array([x[-1] for x in data])

# Importing TESTING data
data_test = np.load("fashion_test.npy")
x_test = np.array([x[:-1] for x in data])/255
y_test = np.array([x[-1] for x in data])

# Train logistic regression model
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Use model to predict on test set
predictions = clf.predict(x_test)

# Evaluation
accuracy_score(y_test, predictions)
confusion_matrix(y_test, predictions)
