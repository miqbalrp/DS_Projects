import cv2
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt 

CATEGORIES = ["NORMAL", "PNEUMONIA"]
model = tf.keras.models.load_model("CNN_chest.model")

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

from keras.utils import to_categorical
y_test = to_categorical(y_test)

n_show = 20

predicted_classes = model.predict(X_test[:n_show])
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

plt.figure()

for i in range (n_show-1):
    plt.subplot(4,5,i+1)
    plt.imshow(X_test[i].reshape(100,100),cmap='gray', interpolation='none')
    plt.title("Pred: {}, Class: {}".format(CATEGORIES[predicted_classes[i]],CATEGORIES[np.argmax(y_test[i])]))
    plt.tight_layout()
    #plt.show()

plt.show()