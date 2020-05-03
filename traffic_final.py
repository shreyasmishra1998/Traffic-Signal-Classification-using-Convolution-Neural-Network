import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import adam

from sklearn.model_selection import train_test_split
EPOCHS = 10
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CATEGORIES = 43
TEST_SIZE = 0.4     
path = []
data = []
label = []

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, label = load_data(sys.argv[1])
    

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(label)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=0.4
    )
    model = get_model()    

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS,validation_data= (x_test, y_test),
              shuffle=True)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)
    

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):

    path = []
    path.append("C:\\Users\\LENOVO\\traffic\\train\\0\\*.*")    
    path.append("C:\\Users\\LENOVO\\traffic\\train\\1\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\2\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\3\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\4\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\5\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\6\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\7\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\8\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\9\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\10\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\11\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\12\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\13\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\14\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\15\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\16\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\17\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\18\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\19\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\20\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\21\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\22\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\23\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\24\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\25\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\26\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\27\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\28\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\29\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\30\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\31\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\32\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\33\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\34\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\35\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\36\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\37\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\38\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\39\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\40\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\41\\*.*")
    path.append("C:\\Users\\LENOVO\\traffic\\train\\42\\*.*")
    for i in range(0,NUM_CATEGORIES):
        for file in glob.glob(path[i]):
            a = cv2.imread(file)
            a = cv2.resize(a,(IMG_WIDTH,IMG_HEIGHT))
            cv2.imwrite(file+'.png', a) 
            data.ap     pend(a)
            label.append(i)
    return(data,label)
    raise NotImplementedError

def get_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=(28,28,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(392))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CATEGORIES))
    model.add(Activation('softmax'))

    opt = adam(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    return (model)
    raise NotImplementedError


if __name__ == "__main__":
    main()
