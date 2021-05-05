"""Script to test image I/O on the GlaS dataset, directly from the files or using the DataGenerator.

Usage python test_dataset.py path_to_dataset

path_to_dataset points to the directory containing all training images and annotations.
"""

import os
import sys

from matplotlib import pyplot as plt
from skimage.io import imread

from DataGenerator import DataGenerator

def direct_io(directory):
    image_files = [os.path.join(directory, f'train_{i}.bmp') for i in range(1, 86)]
    annotation_files = [os.path.join(directory, f'train_{i}_anno.bmp') for i in range(1, 86)]

    for imf,annof in zip(image_files, annotation_files):
        im,anno = imread(imf), imread(annof)
        print(im.shape, im.dtype, anno.shape, anno.dtype)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(im)
        plt.subplot(1,2,2)
        plt.imshow(anno)
        plt.show()
        break

def data_generator(directory):
    dg = DataGenerator(5, 10, directory)

    for batch_x, batch_y in dg.next_batch(1):
        print(batch_x.shape, batch_y.shape, batch_y.min(), batch_y.max())

def main():
    try:
        directory = sys.argv[1]
    except:
        print("path_to_dataset must be provided.")
        sys.exit(1)

    data_generator(directory)