# System

# Third Party
import numpy as np
from keras import backend as K

# Internal
import dvpy as dv
from . import IteratorBase
import os

class NumpyArrayIterator(IteratorBase):
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
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

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
        batch_y1= np.zeros(
            tuple([current_batch_size]) + self.shape + tuple([self.output_channels])
        )
        #batch_y2=np.zeros(tuple([current_batch_size])+(3,))
        #batch_y3=np.zeros(tuple([current_batch_size])+(3,))
        #batch_y4=np.zeros(tuple([current_batch_size])+(3,))
     

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
            patient_id = os.path.dirname(os.path.dirname(self.X[j]))
            affine_path = os.path.join(patient_id,'affine/2C.npy')
            npy_matrix = np.load(affine_path)
            coor_change_path = os.path.join(patient_id,'affine/padding_coordinate_conversion.npy')
            coor_change_matrix = np.load(coor_change_path)

            #convert the coordinate of mpr_center to padding image system:
            volume_center=npy_matrix[0]
            mpr_center=npy_matrix[1]
            volume_center_padding=volume_center+coor_change_matrix
            mpr_center_padding=mpr_center+coor_change_matrix
            
            #also need to read all vectors
            translation_raw=npy_matrix[2]
            translation_n=npy_matrix[3]

            x_raw=npy_matrix[4]
            x_n=npy_matrix[5]
            y_raw=npy_matrix[6]
            y_n=npy_matrix[7]
            n_raw=npy_matrix[8]
            n_n=npy_matrix[9]

            # If *training*, we want to augment the data.
            # If *testing*, we do not.
            if self.augment:
                x, label,translation,rotation,scale,transform_matrix = self.image_data_generator.random_transform(x.astype("float32"), label.astype("float32"))
               
                #translation vector change
                translation_n=dv.tf.change_of_translation_vector_after_augment(volume_center_padding,mpr_center_padding,
                    transform_matrix,adapt_size)
               
                # #x,y directional vector change
                x_n=dv.tf.change_of_direction_vector_after_augment(x_raw,rotation,scale)
                y_n=dv.tf.change_of_direction_vector_after_augment(y_raw,rotation,scale)
                #n_n = dv.tf.change_of_direction_vector_after_augment(n_raw,rotation,scale)
                

            # Normalize the *individual* images to zero mean and unit std
            if self.normalize:
                batch_x[i] = dv.normalize_image(x)
            else:
                batch_x[i] = x

            batch_y1[i] = label
            #batch_y2[i] = translation_n
            #batch_y3[i] = x_n
            #batch_y4[i] = y_n
            
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
                self.image_data_generator.output_layer_names, [batch_y1]
            )
        }
        

        return (inputs, outputs)
