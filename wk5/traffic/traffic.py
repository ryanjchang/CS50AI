import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

EPOCHS = 15
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 11
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    print(sorted(Counter(labels).items()))

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    print("running")

    # Get a compiled neural network
    model = get_model()

    print("got the model")

    # Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2)

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()

    print("fit")

    preds = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test.argmax(axis=1), preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    print("eval")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    labels = []
    images = []
    for entry in os.scandir(data_dir):
        print("scanning entry")
        if entry.is_dir():
            for file in os.scandir(entry.path):
                # not sure if we are supposed to use os.split here
                img = cv2.imread(file.path)
                if img is not None:
                    resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), 0, 0, interpolation = cv2.INTER_AREA)
                    images.append(resized / 255.0) # normalize pixels
                    labels.append(int(entry.name))
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # create a neural network
    model = tf.keras.models.Sequential([
        # specify input shape
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        
        # # transformations
        # tf.keras.layers.RandomRotation(0.05),
        # tf.keras.layers.RandomTranslation(0.1, 0.1),
        # tf.keras.layers.RandomZoom(0.1),
        
        # first input layer
        # 32 filters (usually 32 or 64 at the start to detect edges and stuff)
        # relu 0 if negative or x if positive
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        # goes from 30x30x3 (3 is rgb) to 28x28x32 because edges get cut off and there are 32 filters
        # can keep padding if you want

        # max pooling
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        # run another layer with more units
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        #flatten
        tf.keras.layers.Flatten(),

        #hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # output layer
        # softmax is a multi clsas classification layer -> probability distribution across classes
        # only ever used for output
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # compile the model
    #optimizer = how weights are updated
    # adam - similar to stochastic gradient descent
    # SGD is faster than GD and needs only one sample at a time
    # stochastic means there's randomness
    # adam tracks the mean of recent gradients (momentum)
    # and the mean of recent squared gradients
    # loss = which loss is minimized
    # since we are trying to minimize categorical errors, we use categorical cross entropy
    # metrics = what you are tracking
    # accurace = correct predictions
    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    main()
