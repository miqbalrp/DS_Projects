import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import *

import matplotlib.pyplot as plt

# import data from pickle file
import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

# just using a part of data
#X = X[:int(0.6*len(X))]
#y = y[:int(0.6*len(y))]

# if using categorical_crossentropy loss
from keras.utils import to_categorical
y = to_categorical(y)

# normalization
X = X/255.0

# build network
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(50, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax'))
model.summary()

# set hyperparameter
batch_size = 100
epochs = 10
validation_split = 0.2


# compile the model
optimizer = Adam(lr=1e-3)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# fit to training data
training = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

# evaluate on test data
#model.evaluate(test_data, test_labels, batch_size=10)

# extract the history f
history = training.history

# plot the training loss and validation loss
plt.figure(1)
plt.title('Training loss and validation loss')
plt.plot(history['loss'], label='training')
plt.plot(history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# plot the training accuracy and validation accuracy
plt.figure(2)
plt.title('Training accuracy and validation accuracy')
plt.plot(history['acc'], label='training')
plt.plot(history['val_acc'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# save model network
model.save('CNN_chest.model')
print("Model saved as CNN_chest.model")
model.summary()

