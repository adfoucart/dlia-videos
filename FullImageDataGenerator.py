import os

import numpy as np
from skimage.transform import resize

from DataGenerator import DataGenerator

class FullImageDataGenerator(DataGenerator):
    """Reads dataset files (images & annotations) and prepare training & validation batches

    Uses full images (no tiling)"""

    def __init__(self, batch_size, validation_size, directory, image_size):
        super().__init__(batch_size, validation_size, directory)

        self.image_size = image_size
        self.batches_per_epoch = len(self.train_idxs)//self.batch_size

        self.val_x = np.array([self.preprocess_image(self.full_images[idx]) for idx in self.val_idxs])
        self.val_y = np.array([self.preprocess_anno(self.full_annotations[idx]) for idx in self.val_idxs])

        self.train_x = np.array([self.preprocess_image(self.full_images[idx]) for idx in self.train_idxs])
        self.train_y = np.array([self.preprocess_anno(self.full_annotations[idx]) for idx in self.train_idxs])

    def next_batch(self, n_epochs):
        """Generator of tuples of (image batch, annotation batch).
        Randomly shuffles the order of the images between each epoch.
        """
        for e in range(n_epochs):
            np.random.shuffle(self.train_idxs)
            for i in range(self.batches_per_epoch):
                batch_x = self.train_x[i*self.batch_size:(i+1)*self.batch_size]
                batch_y = self.train_y[i*self.batch_size:(i+1)*self.batch_size]>0
                yield self._augment(batch_x,batch_y)

    def get_validation_data(self):
        """Returns all the data set aside in the constructor for validation as a tuple of (images, annotations)"""
        return self.val_x, self.val_y>0

    def get_validation_data_labels(self):
        return self.val_x, self.val_y

    def preprocess_image(self, im):
        return resize(im, self.image_size)

    def preprocess_anno(self, anno):
        return resize(anno, self.image_size, preserve_range=True, order=0, anti_aliasing=False)        