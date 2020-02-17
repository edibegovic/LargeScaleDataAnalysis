
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras import activations
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import Sequential
import numpy as np

# Importing TRAINING data
data = np.load("fashion_train.npy")
x_train = np.array([x[:-1] for x in data])/255
y_train = np.array([x[-1] for x in data])

# Importing TESTING data
data_test = np.load("fashion_test.npy")
x_test = np.array([x[:-1] for x in data])/255
y_test = np.array([x[-1] for x in data])

# One-hot-like encoding
y_train = keras.utils.to_categorical(y_train, 5)
y_test = keras.utils.to_categorical(y_test, 5)

# Sequential stack of perceptrons (MLP model)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

# Overview of model
model.summary()

# Initialize optimizer and its metrics
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train,
                    batch_size = 128,
                    epochs = 20,
                    validation_data=(x_test, y_test))

# Evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])
