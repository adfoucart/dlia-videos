import os

import numpy as np
from skimage.io import imread
from skimage.exposure import adjust_gamma

class DataGenerator():
    """Reads dataset files (images & annotations) and prepare training & validation batches"""

    def __init__(self, batch_size, validation_size, directory, train_test='train'):
        """Load dataset into RAM & prepare train/validation split

        Reads train_%d.bmp & train_%d_anno.bmp files in directory and randomly set aside 
        validation_size images from the dataset for validation.
        """
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.directory = directory

        nPerSet = {'train': 85, 'testA': 60, 'testB': 20}
        self.image_files = [os.path.join(self.directory, f'{train_test}_{i}.bmp') for i in range(1, nPerSet[train_test]+1)]
        self.annotation_files = [os.path.join(self.directory, f'{train_test}_{i}_anno.bmp') for i in range(1, nPerSet[train_test]+1)]

        # Pre-load all images in RAM
        self.full_images = [imread(f)/255 for f in self.image_files]
        self.full_annotations = [imread(f) for f in self.annotation_files]

        # Train/Validation split
        self.idxs = np.arange(len(self.image_files))
        np.random.seed(1)
        np.random.shuffle(self.idxs)

        self.val_idxs = self.idxs[:self.validation_size]
        self.train_idxs = self.idxs[self.validation_size:]

    @staticmethod
    def _augment(batch_x, batch_y):
        """Basic data augmentation:
        Horizontal/Vertical symmetry
        Random noise
        Gamma correction"""

        # Vertical symmetry
        if( np.random.random()<0.5 ):
            batch_x = batch_x[:,::-1,:,:]
            batch_y = batch_y[:,::-1,:]
        # Horizontal symmetry
        if( np.random.random()<0.5 ):
            batch_x = batch_x[:,:,::-1,:]
            batch_y = batch_y[:,:,::-1]

        # Gamma (before random noise because input values must be between [0,1])
        gamma = (np.random.random()-0.5)*2
        if gamma < 0:
            gamma=1/(1-gamma)
        else:
            gamma=1+gamma
        batch_x_ = batch_x.copy()
        for i in range(len(batch_x)):
            batch_x_[i] = adjust_gamma(batch_x[i], gamma=gamma)
        
        # Random noise
        batch_x_ += np.random.normal(0, 0.02, size=batch_x.shape)

        return batch_x_,batch_y