import sys

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from FullImageDataGenerator import FullImageDataGenerator
from TileDataGenerator import TileDataGenerator
from BaseModel import BaseModel
from ShortSkipModel import ShortSkipModel
from LongSkipModel import LongSkipModel
from Evaluator import Evaluator

def train_experiment(experiment):
    '''Train for an experiment, and compute metrics on training and validation set.

    experiment is a dictionary with three required entries:
        - datagen -> the DataGenerator class
        - model -> the Model class
        - clf_name -> name of the classifier, which will be used to set the saved file names''' 
    try:
        DataGenerator = experiment['datagen']
        Model = experiment['model']
        clf_name = experiment['name']
    except KeyError:
        print("Experiment dictionary must be contain dataget, model and name keys.")

    # optional parameters
    lr = experiment.get('lr', 1e-4)
    eps = experiment.get('eps', 1e-8)
    overlap = experiment.get('overlap', 'minimum')
    max_epochs = experiment.get('max_epochs', 200)
    patience = experiment.get('patience', 15)
    directory = experiment.get('directory', "D:/Adrien/dataset/GlaS/train")
    image_size = experiment.get('image_size', (256,384))
    batch_size = experiment.get('batch_size', 5)
    validation_size = experiment.get('validation_size', 10)
    min_area = experiment.get('min_area', 4000)
    
    tf.keras.backend.clear_session()

    # Training
    generator = DataGenerator(batch_size, validation_size, directory, image_size)
    model = Model(image_size, clf_name, lr=lr, eps=eps)
    history = model.fit(max_epochs, generator, patience=patience)

    model.save(f"{clf_name}.hdf5")

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(history.history['loss'], 'b-')
    plt.subplot(2,1,2)
    plt.plot(history.history['accuracy'], 'b-')
    plt.savefig(f'{clf_name}_history.png')

def main():
    try:
        directory = sys.argv[1]
    except:
        raise ValueError("No directory provided. Usage: python main_experiment.py path_to_train_dataset")

    experiments = [
        {'datagen' : FullImageDataGenerator,
        'model' : LongSkipModel,
        'name' : 'long_skip_model_full_image_retrain',
        'lr' : 1e-3,
        'eps': 1e-7,
        'directory': directory,
        'max_epochs': 500,
        'patience': 500,
        'validation_size': 0}
        ]

    for experiment in experiments:
        print(f"Training {experiment['name']}")
        train_experiment(experiment)

if __name__ == '__main__':
    main()