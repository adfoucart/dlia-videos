import numpy as np
import tensorflow as tf
from DataGenerator import DataGenerator
from Model import Model
from matplotlib import pyplot as plt

'''
### Data generator
def generate_data(batch_size, n_epochs, batches_per_epoch):
	for e in range(n_epochs):
		for b in range(batches_per_epoch):
			batch_x = np.random.random((batch_size, 256, 256, 3))
			batch_y = np.random.random((batch_size, 256, 256))>0.85
			yield batch_x,batch_y

### Model construction
def get_model():
	inputs = tf.keras.Input(shape=(256,256,3))
	x = tf.keras.layers.Conv2D(16, 3, activation=tf.nn.relu, padding='same')(inputs)
	outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax)(x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)

	model.compile(optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
	return model

### Model fit
def fit_model(model, generator, n_epochs, batches_per_epoch):
	model.fit(generator, epochs=n_epochs, steps_per_epoch=batches_per_epoch)


### Evaluation
def evaluate_model(model, validation_data):
	pred = np.argmax(model.predict(validation_data[0]), axis=3)
	acc = (pred==validation_data[1]).sum()/(np.prod(pred.shape))
	return acc
'''

if __name__ == '__main__':
	import sys

	loadFrom = None if len(sys.argv) <= 1 else sys.argv[1]

	batch_size = 5
	n_epochs = 100
	batches_per_epoch = 10
	
	dataset = DataGenerator(batch_size, 10, "d:/Adrien/dataset/GlaS/train")
	model = Model(loadFrom=loadFrom)#loadFrom='model_1.hdf5')
	model.print()
	model.plot()

	if( loadFrom == None ):
		history = model.fit(n_epochs, dataset)
		model.save('model_1.hdf5')

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