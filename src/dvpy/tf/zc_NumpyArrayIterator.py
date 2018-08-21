# Third Party
import numpy as np
from keras import backend as K
import os

# Internal
import dvpy as dv
from . import IteratorBase
import dvpy.tf


class zc_NumpyArrayIterator(IteratorBase):
    def __init__(
        self,
        X,
        y,
        image_data_generator,
        batch_size=32,
        shuffle=False,
        seed=None,
        input_adapter=None,
        output_adapter=None,
        shape=None,
        input_channels=None,
        output_channels=None,
        augment=False,
        normalize=False,
    ):

        if K.image_dim_ordering() != "tf":
            raise Exception("Only tensorflow backend is supported.")

        if len(X) != len(y):

            raise Exception(
                "X (images tensor) and y (labels) "
                "should have the same length. "
                "Found: X.shape = %s, y.shape = %s"
                % (np.asarray(X).shape, np.asarray(y).shape)
            )
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.shape = shape
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.augment = augment
        self.normalize = normalize
        super(zc_NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        ##
        ## Allocate Memory
        ##

        # Input Image in range 0 to 1
        batch_x = np.zeros(
            tuple([current_batch_size]) + self.shape + tuple([self.input_channels])
        )

        # Ground Truth Segmentation
        batch_y1 = np.zeros(
            tuple([current_batch_size]) + self.shape + tuple([self.output_channels])
        )
        batch_y2=np.zeros(tuple([current_batch_size])+(3,))
        

        ##
        ## Deal with Actual Data
        ##

        # index_array is a randomly shuffled list of cases for this batch
        for i, j in enumerate(index_array):

            # Retrieve the path to the input image...
            x = self.X[j]
            # ...and convert the path to a raw (unnormalized) image.
            if self.input_adapter is not None:
                x = self.input_adapter(x)
            adapt_size=x.shape

            # Retrieve the path to the segmentation...
            label = self.y[j]
            if self.output_adapter is not None:
                # ...and convert the path to a one-hot encoded image.
                label = self.output_adapter(label)
            
            #Retrieve the path to the matrix npy file (the original translation vector)
            #patient_id=os.path.dirname(os.path.dirname(self.X[j]))
            #npy_path=os.path.join(patient_id,'matrix/2C.npy')
            #npy_matrix=np.load(npy_path)
            #translation_vector=npy_matrix[0,:]
            
            # If *training*, we want to augment the data.
            # If *testing*, we do not.
            if self.augment:
                x, label,_,rotation,scale,_ = self.image_data_generator.random_transform(
                    x.astype("float32"), label.astype("float32")
                )
                #translation_vector=dvpy.tf.change_of_translation_after_transform(translation_vector,rotation,scale,adapt_size)
            
            

            # Normalize the *individual* images to zero mean and unit std
            if self.normalize:
                batch_x[i] = dv.normalize_image(x)
            batch_y1[i] = label
            #batch_y2[i] =translation_vector
            batch_y=[batch_y1]
           # batch_y=[batch_y1,batch_y2]
            

        ##
        ## Return
        ##

        inputs = {
            name: layer
            for name, layer in zip(
                self.image_data_generator.input_layer_names, [batch_x]
            )
        }
        outputs = {
            name: layer
            for name, layer in zip(
                self.image_data_generator.output_layer_names, batch_y
            )
        }

        return (inputs, outputs)
