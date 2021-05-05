import tensorflow as tf
from skimage.measure import label, regionprops
import numpy as np

class Model():
    """Build & use DCNN model.
    Includes post-processing.
    """

    def __init__(self, image_size, clf_name, loadFrom=None):
        """Load existing model from hdf5 file or build it from scratch."""
        self.image_size = image_size
        self.clf_name = clf_name
        if( loadFrom == None ):
            self.set_model()
            opt = tf.keras.optimizers.Adam(
                learning_rate=1e-4, 
                epsilon=1e-8,
                name='Adam')
            self.model.compile(
                optimizer=opt, 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                metrics=[tf.keras.losses.SparseCategoricalCrossentropy(name='crossentropy'), 'accuracy']
                )
        else:
            self.model = tf.keras.models.load_model(loadFrom, compile=False, custom_objects={'leaky_relu': tf.nn.leaky_relu})

    def print(self):
        """Display model summary"""
        self.model.summary()

    def plot(self):
        """Generates plot of model architecture and saves it to model.png file"""
        tf.keras.utils.plot_model(self.model, show_shapes=True)

    def save(self, fname):
        """Save whole model to file"""
        self.model.save(fname)

    def fit(self, n_epochs, dataset, patience=15):
        """Fit the model on the dataset with EarlyStopping on the validation crossentropy"""
        return self.model.fit(
            dataset.next_batch(n_epochs), 
            epochs=n_epochs, 
            steps_per_epoch=dataset.batches_per_epoch, 
            validation_data=dataset.get_validation_data(), 
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_crossentropy', patience=patience),
                tf.keras.callbacks.ModelCheckpoint(f"{self.clf_name}.hdf5", save_best_only=True)
                ]
            )

    def predict(self, data):
        """Output will be batch_sizex256x256x2, with [:,:,:,1] = p(gland)"""
        return self.model.predict(data)

    @staticmethod
    def post_process(pred, min_area=250):
        """Label binary mask, then remove small objects & close holes"""
        pred_mask = np.argmax(pred, axis=2)
        lab = label(pred_mask)
        for obj in regionprops(lab):
            if( obj.area < min_area ):
                lab[lab==obj.label] = 0
            else:
                region = lab[obj.bbox[0]:obj.bbox[2],obj.bbox[1]:obj.bbox[3]]
                region[obj.filled_image] = obj.label
        
        return lab