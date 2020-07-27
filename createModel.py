import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import numpy as np

from model import build_simpleNet
mnist = tf.keras.datasets.mnist

#loading mnist datasets.
#if this is the first time downloading mnist, it will takes a few minutes.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#prepocessing
x_train, x_test = x_train / 255, x_test / 255
x_train[x_train != 0 ] = 1

model = build_simpleNet()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model for prediction and save it.
history = model.fit(x_train, y_train, epochs=5)
model.save('mnist.model')

print('model is created!!')

