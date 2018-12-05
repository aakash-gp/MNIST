from keras.models import load_model
import numpy as np
import cv2
	
# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale


model1 = load_model('modeldnn.h5')
model2 = load_model('modelrnn.h5')
model3 = load_model('modelcnn.h5')

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.show()
classes = model1.predict_classes(np.reshape(X_train[0],[1, 1,28,28]))
print ("DNN: ", classes)
classes = model2.predict_classes(np.reshape(X_train[0],[1,28,28]))
print ("RNN: ", classes)
classes = model3.predict_classes(np.reshape(X_train[0],[1,1,28,28]))
print ("CNN: ", classes, "\n")



plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.show()
classes = model1.predict_classes(np.reshape(X_train[1],[1,1,28,28]))
print ("DNN: ", classes)
classes = model2.predict_classes(np.reshape(X_train[1],[1,28,28]))
print ("RNN: ", classes)
classes = model3.predict_classes(np.reshape(X_train[1],[1,1,28,28]))
print ("CNN: ", classes, "\n")



plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.show()
classes = model1.predict_classes(np.reshape(X_train[2],[1,1,28,28]))
print ("DNN: ", classes)
classes = model2.predict_classes(np.reshape(X_train[2],[1,28,28]))
print ("RNN: ", classes)
classes = model3.predict_classes(np.reshape(X_train[2],[1,1,28,28]))
print ("CNN: ", classes, "\n")


plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.show()
classes = model1.predict_classes(np.reshape(X_train[3],[1,1,28,28]))
print ("DNN: ", classes)
classes = model2.predict_classes(np.reshape(X_train[3],[1,28,28]))
print ("RNN: ", classes)
classes = model3.predict_classes(np.reshape(X_train[3],[1,1,28,28]))
print ("CNN: ", classes, "\n")
