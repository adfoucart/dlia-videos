"""Script to test the training and validation pipeline.

Usage: python pipeline.py path_to_dataset [path_to_model]

path_to_dataset : path to the directory containing all training images and annotations 
of the GlaS challenge dataset.

If path_to_model is omitted: train a network on the GlaS dataset, plot the loss & accuracy.
If path_to_model is set: load model.

Save trained model to model.hdf5 or first non-existent model_%d.hdf5
Plot predictions on a few images from the training set.

Contains methods to generate fake data & model to test the basic keras functionalities.
"""

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from DataGenerator import DataGenerator
from Model import Model

def generate_data(batch_size, n_epochs, batches_per_epoch):
    """Generate fake random data"""
    for e in range(n_epochs):
        for b in range(batches_per_epoch):
            batch_x = np.random.random((batch_size, 256, 256, 3))
            batch_y = np.random.random((batch_size, 256, 256))>0.85
            yield batch_x,batch_y

def get_model():
    """Create minimal convolutional model"""
    inputs = tf.keras.Input(shape=(256,256,3))
    x = tf.keras.layers.Conv2D(16, 3, activation=tf.nn.relu, padding='same')(inputs)
    outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

def fit_model(model, generator, n_epochs, batches_per_epoch):
    """Fit model with data generator
    
    model : tf.keras.Model
    generator : yield (batch_x, batch_y)
    """
    model.fit(generator, epochs=n_epochs, steps_per_epoch=batches_per_epoch)


def evaluate_model(model, validation_data):
    """Compute accuracy of the model on the validation data.

    model : tf.keras.Model
    validation_data : tuple (data_x, data_y)"""
    pred = np.argmax(model.predict(validation_data[0]), axis=3)
    acc = (pred==validation_data[1]).sum()/(np.prod(pred.shape))
    return acc

def usage():
    print("Usage: python pipeline.py path_to_dataset [path_to_model]")

def main():
    try:
        path_to_dataset = sys.argv[1]
    except:
        print("path_to_dataset must be provided.")
        usage()
        sys.exit(1)

    # optional parameter
    loadFrom = None if len(sys.argv) <= 2 else sys.argv[2]

    batch_size = 5
    n_epochs = 100
    batches_per_epoch = 10
    image_size = (256,384)
    
    dataset = DataGenerator(batch_size, 10, path_to_dataset, image_size)
    model = Model(image_size, loadFrom=loadFrom)
    model.print()
    model.plot()

    if( loadFrom == None ):
        history = model.fit(n_epochs, dataset)
        path_to_model = 'model.hdf5'
        i = 1
        while( os.path.exists(path_to_model) ):
            path_to_model = f'model_{i}.hdf5'
            i += 1
        model.save(path_to_model)

        plt.figure()
        plt.plot(history.history['loss'], 'b-', label='training')
        plt.plot(history.history['val_loss'], 'r-', label='validation')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()

        plt.figure()
        plt.plot(history.history['accuracy'], 'b-', label='training')
        plt.plot(history.history['val_accuracy'], 'r-', label='validation')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    ## Plot a few results from the training data:
    batch_x = dataset.images[dataset.train_idxs[:5]]
    batch_y = dataset.annotations[dataset.train_idxs[:5]]
    pred = model.predict(batch_x)
    pred_mask = np.argmax(pred, axis=3)

    for i in range(5):
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(batch_x[i])
        plt.contour(batch_y[i])
        plt.subplot(1,3,2)
        plt.imshow(pred[i,:,:,1])
        plt.subplot(1,3,3)
        plt.imshow(pred_mask[i])
        plt.show()

if __name__ == '__main__':
    main()