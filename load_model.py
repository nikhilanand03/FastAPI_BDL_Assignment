from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
num_classes = 10
x_train = X_train.reshape(60000, 784)
x_test = X_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape, 'train input samples')
print(x_test.shape, 'test input samples')

y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)
print(y_train.shape, 'train output samples')
print(y_test.shape, 'test output samples')

model2 = keras.Sequential()
model2.add(layers.Dense(256, activation='sigmoid', input_shape=(784,)))
model2.add(layers.Dense(128, activation='sigmoid'))
model2.add(layers.Dense(10, activation='softmax'))
model2.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model2.load_weights("/Users/nikhilanand/FastAPI_BDL/training_1/cp.weights.h5")
# loss, acc = model2.evaluate(x_test, y_test, verbose=2)
# print("Test accuracy: {:5.2f}%".format(100*acc))
# loss, acc = model2.evaluate(x_train, y_train, verbose=2)
# print("Train accuracy: {:5.2f}%".format(100*acc))

print(np.argmax(model2(np.array(x_test[0]).reshape(1,784))))