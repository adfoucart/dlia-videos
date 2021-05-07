import tensorflow as tf

from Model import Model

class BaseModel(Model):
    """Build & use DCNN model.
    Includes post-processing.
    """

    def __init__(self, image_size, clf_name, loadFrom=None, lr=1e-4, eps=1e-8):
        super().__init__(image_size, clf_name, loadFrom=loadFrom, lr=lr, eps=eps)

    def set_model(self):
        inputs = tf.keras.Input(shape=self.image_size+(3,))
        x = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.leaky_relu, padding='same')(inputs)
        x = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.leaky_relu, padding='same')(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        x = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(x)
        x = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        x = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')(x)
        x = tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.leaky_relu, padding='same')(x)
        outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)