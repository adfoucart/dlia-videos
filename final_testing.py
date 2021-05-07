import sys
import os 

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from FullImageDataGenerator import FullImageDataGenerator
from TileDataGenerator import TileDataGenerator
from BaseModel import BaseModel
from ShortSkipModel import ShortSkipModel
from LongSkipModel import LongSkipModel
from Evaluator import Evaluator

def test_experiment(experiment):
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

    model = Model(image_size, clf_name, loadFrom=f'{clf_name}.hdf5')
    
    # Test A
    # Compute & save metrics
    generator = DataGenerator(batch_size, validation_size, os.path.join(directory, 'testA'), image_size, train_test='testA')
    metrics = Evaluator.evaluate(model, generator, 'train', overlap, min_area)
    
    with open(f"{clf_name}_testA_metrics.txt", 'w') as fp:
        print("Test A")
        print("Precision\tRecall\tMCC", file=fp)
        print(metrics.mean(axis=0), file=fp)
        print(np.median(metrics,axis=0), file=fp)
        
    np.save(f"{clf_name}_testA_metrics.npy", metrics)

    generator = DataGenerator(batch_size, validation_size, os.path.join(directory, 'testB'), image_size, train_test='testB')
    metrics = Evaluator.evaluate(model, generator, 'train', overlap, min_area)
    
    with open(f"{clf_name}_testB_metrics.txt", 'w') as fp:
        print("Test B")
        print("Precision\tRecall\tMCC", file=fp)
        print(metrics.mean(axis=0), file=fp)
        print(np.median(metrics,axis=0), file=fp)
        
    np.save(f"{clf_name}_testB_metrics.npy", metrics)


def main():
    try:
        directory = sys.argv[1]
    except:
        raise ValueError("No directory provided. Usage: python main_experiment.py path_to_train_dataset")

    experiments = [
        {'datagen' : FullImageDataGenerator,
        'model' : LongSkipModel,
        'name' : 'long_skip_model_full_image_retrain',
        'directory': directory,
        'validation_size': 0,
        'min_area': 2000}
        ]

    for experiment in experiments:
        print(f"Test {experiment['name']}")
        test_experiment(experiment)

if __name__ == '__main__':
    main()