from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop
from keras.utils import np_utils



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# reshape to be [samples][pixels][width][height]
x_train = x_train.reshape(x_train.shape[0], -1, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], -1, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


print('Evaluate IRNN...')

model = Sequential()
#model.add(SimpleRNN(hidden_units, kernel_initializer=initializers.RandomNormal(stddev=0.001), recurrent_initializer=initializers.Identity(gain=1.0), activation='relu', input_shape=x_train.shape[1:]))
model.add(SimpleRNN(hidden_units, activation='relu', input_shape=x_train.shape[1:]))

model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)

scores = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("RNN Error: %.2f%%" % (100-scores[1]*100))

model.save('modelrnn.h5')

model.summary()