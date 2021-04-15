import numpy as np
import tensorflow as tf
from DataGenerator import DataGenerator
from Model import Model

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
	batch_size = 5
	n_epochs = 2
	batches_per_epoch = 10
	
	dataset = DataGenerator(batch_size, batches_per_epoch, 10)
	model = Model()
	model.print()

	model.fit(n_epochs, dataset)