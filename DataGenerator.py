import numpy as np

class DataGenerator():

	def __init__(self, batch_size, batches_per_epoch, validation_size):
		self.batch_size = batch_size
		self.batches_per_epoch = batches_per_epoch
		self.validation_size = validation_size

		self.val_x = np.random.random((10, 256, 256, 3))
		self.val_y = np.random.random((10,256,256))>0.85

	def next_batch(self, n_epochs):
		for e in range(n_epochs):
			for i in range(self.batches_per_epoch):
				batch_x = np.random.random((self.batch_size, 256, 256, 3))
				batch_y = np.random.random((self.batch_size, 256, 256))>0.85
				yield batch_x,batch_y

	def get_validation_data(self):
		return self.val_x, self.val_y