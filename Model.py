import tensorflow as tf

class Model():

	def __init__(self, loadFrom=None):
		if( loadFrom == None ):
			self.set_model()
			self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.losses.SparseCategoricalCrossentropy(name='crossentropy'), 'accuracy'])
		else:
			self.model = tf.keras.models.load_model(loadFrom, compile=False)

	def set_model(self):
		# Model definition
		#	Input 		[256x256x3]
		#	Conv2D 		[256x256x32]
		#	Conv2D 		[256x256x32]
		# 	MaxPool2D 	[128x128x32]
		#	Conv2D		[128x128x64]
		# 	Conv2D		[128x128x64]
		#	MaxPool2D 	[64x64x64]
		# 	Conv2D		[64x64x128]
		#	Conv2D		[64x64x128]
		#	UpSampling 	[128x128x128]
		#	Conv2D 		[128x128x64]
		#	UpSampling 	[256x256x64]
		#	Conv2D 		[256x256x32]
		#	Ouptuts 	[256x256x2]
		inputs = tf.keras.Input(shape=(256,256,3))
		x = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu, padding='same')(inputs)
		x = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu, padding='same')(x)
		x = tf.keras.layers.MaxPool2D(2)(x)
		x = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu, padding='same')(x)
		x = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu, padding='same')(x)
		x = tf.keras.layers.MaxPool2D(2)(x)
		x = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.relu, padding='same')(x)
		x = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.relu, padding='same')(x)
		x = tf.keras.layers.UpSampling2D(2)(x)
		x = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu, padding='same')(x)
		x = tf.keras.layers.UpSampling2D(2)(x)
		x = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu, padding='same')(x)
		outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax)(x)

		self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

	def print(self):
		self.model.summary()

	def plot(self):
		tf.keras.utils.plot_model(self.model, show_shapes=True)

	def save(self, fname):
		self.model.save(fname)

	def fit(self, n_epochs, dataset):
		return self.model.fit(dataset.next_batch(n_epochs), epochs=n_epochs, steps_per_epoch=dataset.batches_per_epoch, validation_data=dataset.get_validation_data(), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_crossentropy', patience=15)])

	def predict(self, data):
		return self.model.predict(data)