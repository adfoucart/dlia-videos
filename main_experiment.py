import sys

from matplotlib import pyplot as plt
import tensorflow as tf

from FullImageDataGenerator import FullImageDataGenerator
from TileDataGenerator import TileDataGenerator
from BaseModel import BaseModel
from ShortSkipModel import ShortSkipModel
from LongSkipModel import LongSkipModel
from Evaluator import Evaluator

def train_experiment(experiment):
    DataGenerator = experiment['datagen']
    Model = experiment['model']
    clf_name = experiment['name']

    batch_size = 5
    validation_size = 10
    directory = "D:/Adrien/dataset/GlaS/train"
    image_size = (256,384)
    max_epochs = 200

    tf.keras.backend.clear_session()

    generator = DataGenerator(batch_size, validation_size, directory, image_size)
    model = Model(image_size, clf_name)
    history = model.fit(max_epochs, generator)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(history.history['val_loss'], 'r-')
    plt.plot(history.history['loss'], 'b-')
    plt.subplot(2,1,2)
    plt.plot(history.history['val_accuracy'], 'r-')
    plt.plot(history.history['accuracy'], 'b-')
    plt.savefig(f'{clf_name}_history.png')

def main():
    experiments = [
        {'datagen' : FullImageDataGenerator,
        'model' : BaseModel,
        'name' : 'base_model_full_image'},
        {'datagen' : FullImageDataGenerator,
        'model' : ShortSkipModel,
        'name' : 'short_skip_model_full_image'},
        {'datagen' : FullImageDataGenerator,
        'model' : LongSkipModel,
        'name' : 'long_skip_model_full_image'},
        {'datagen' : TileDataGenerator,
        'model' : BaseModel,
        'name' : 'base_model_tile'},
        {'datagen' : TileDataGenerator,
        'model' : ShortSkipModel,
        'name' : 'short_skip_model_tile'},
        {'datagen' : TileDataGenerator,
        'model' : LongSkipModel,
        'name' : 'long_skip_model_tile'}
        ]

    for experiment in experiments:
        print(f"Training {experiment['name']}")
        train_experiment(experiment)

if __name__ == '__main__':
    main()