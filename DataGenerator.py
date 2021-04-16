import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os

class DataGenerator():

	def __init__(self, batch_size, validation_size, directory):
		self.batch_size = batch_size
		self.validation_size = validation_size
		self.directory = directory

		self.image_files = [os.path.join(self.directory, f'train_{i}.bmp') for i in range(1, 86)]
		self.annotation_files = [os.path.join(self.directory, f'train_{i}_anno.bmp') for i in range(1, 86)]

		# Pre-load all images in RAM
		self.images = np.array([DataGenerator.__preprocess(imread(f)) for f in self.image_files])
		self.annotations = np.array([DataGenerator.__preprocess(imread(f)) for f in self.annotation_files])

		# Train/Validation split
		self.idxs = np.arange(len(self.image_files))
		np.random.seed(1)
		np.random.shuffle(self.idxs)

		self.val_idxs = self.idxs[:self.validation_size]
		self.train_idxs = self.idxs[self.validation_size:]

		self.val_x = self.images[self.val_idxs]
		self.val_y = self.annotations[self.val_idxs]

		self.train_x = self.images[self.train_idxs]
		self.train_y = self.annotations[self.train_idxs]

		self.batches_per_epoch = len(self.train_idxs)//self.batch_size

	def __preprocess(im):
		return resize(im, (256,256))

	def next_batch(self, n_epochs):
		for e in range(n_epochs):
			for i in range(self.batches_per_epoch):
				batch_x = self.train_x[i*self.batch_size:(i+1)*self.batch_size]
				batch_y = self.train_y[i*self.batch_size:(i+1)*self.batch_size]>0
				yield batch_x,batch_y
			np.random.shuffle(self.train_idxs)

	def get_validation_data(self):
		return self.val_x, self.val_y>0