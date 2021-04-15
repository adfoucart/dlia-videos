import tensorflow as tf

class Model():

	def __init__(self):
		inputs = tf.keras.Input(shape=(256,256,3))
		x = tf.keras.layers.Conv2D(16, 3, activation=tf.nn.relu, padding='same')(inputs)
		outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax)(x)
		self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

		self.model.compile(optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

	def print(self):
		self.model.summary()

	def fit(self, n_epochs, dataset):
		self.model.fit(dataset.next_batch(n_epochs), epochs=n_epochs, steps_per_epoch=dataset.batches_per_epoch, validation_data=dataset.get_validation_data())

	def predict(self, data):
		return self.model.predict(data)