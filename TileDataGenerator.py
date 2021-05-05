import os
import math

import numpy as np
from skimage.transform import resize

from DataGenerator import DataGenerator

class TileDataGenerator(DataGenerator):
    """Reads dataset files (images & annotations) and prepare training & validation batches

    Uses tiled images"""

    def __init__(self, batch_size, validation_size, directory, tile_size):
        super().__init__(batch_size, validation_size, directory)

        self.tile_size = tile_size
        self.batches_per_epoch = len(self.train_idxs)

    def next_batch(self, n_epochs):
        """Generator of tuples of (image batch, annotation batch).
        Randomly shuffles the order of the images between each epoch.
        Randomly samples tiles within an image.
        """
        for e in range(n_epochs):
            np.random.shuffle(self.train_idxs)
            for idx in self.train_idxs:
                im = self.full_images[idx]
                anno = self.full_annotations[idx]>0
                batch_x, batch_y = self._get_tiles(im,anno)
                yield self._augment(batch_x,batch_y)

    def get_validation_data(self, labels=False):
        """Returns regular tiles from the validation data as a tuple of (images, annotations)"""
        # val_x = np.zeros((tiles_per_image*len(self.val_idxs),)+self.tile_size+(3,))
        # val_y = np.zeros((tiles_per_image*len(self.val_idxs),)+self.tile_size)
        val_x = []
        val_y = []

        for i,idx in enumerate(self.val_idxs):
            im = self.full_images[idx]
            anno = self.full_annotations[idx] if labels else self.full_annotations[idx]>0
            
            tiles = self._get_regular_tiling(im.shape)
            for tx,ty in tiles:
                tiles_x += [im[ty:ty+self.tile_size[0],tx:tx+self.tile_size[1]]]
                tiles_y += [anno[ty:ty+self.tile_size[0],tx:tx+self.tile_size[1]]]

        return np.array(val_x), np.array(val_y)

    def get_validation_data_labels(self):
        return self.get_validation_data(labels=True)

    def stitch(self, tiles_prediction, imshape):
        tiles = self._get_regular_tiling(imshape)

        pred_image = np.zeros(imshape+(2,)).astype('float')
        n_preds = np.zeros(imshape).astype('float') # to keep track of how many predictions were made on a given pixel

        for i,tile in enumerate(tiles):
            pred = tiles_prediction[i]

            tx = tile[0]
            ty = tile[1]
            
            pred_image[ty:ty+self.tile_size[0],tx:tx+self.tile_size[1],0] += pred[:,:,0]
            n_preds[ty:ty+self.tile_size[0],tx:tx+self.tile_size[1]] += 1
                
        pred_image[:,:,0] /= n_preds
        pred_image[:,:,1] = 1-pred_image[:,:,0]

        return pred_image

    def _get_tiles(self, im, anno):
        pos_y = np.random.randint(0, im.shape[0]-self.tile_size[0], size=(self.batch_size,))
        pos_x = np.random.randint(0, im.shape[1]-self.tile_size[1], size=(self.batch_size,))

        batch_x = np.zeros((self.batch_size,)+self.tile_size+(3,))
        batch_y = np.zeros((self.batch_size,)+self.tile_size)
        for i,pos in enumerate(zip(pos_y,pos_x)):
            batch_x[i] = im[pos[0]:pos[0]+self.tile_size[0], pos[1]:pos[1]+self.tile_size[1]]
            batch_y[i] = anno[pos[0]:pos[0]+self.tile_size[0], pos[1]:pos[1]+self.tile_size[1]]

        return batch_x, batch_y

    def _get_regular_tiling(self, imshape):
        ny = math.ceil(imshape[0]/self.tile_size[0])
        nx = math.ceil(imshape[1]/self.tile_size[1])
        step_y = (imshape[0]-self.tile_size[0])/(ny-1)
        step_x = (imshape[1]-self.tile_size[1])/(nx-1)
        coords_y = np.arange(0, imshape[0]-self.tile_size[0]+1, step_y).astype('int')
        coords_x = np.arange(0, imshape[1]-self.tile_size[1]+1, step_x).astype('int')
        mesh = np.meshgrid(coords_x,coords_y)
        return zip(mesh[0].flatten(), mesh[1].flatten())