# System

# Third Party
import numpy as np
from keras import backend as K
import sympy as sym

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
        batch_y2=np.zeros(tuple([current_batch_size])+(3,))
        batch_y3=np.zeros(tuple([current_batch_size])+(3,))
        batch_y4=np.zeros(tuple([current_batch_size])+(3,))
     

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
            affine_path = os.path.join(patient_id,'affine/BASAL_new.npy')
            M = np.load(affine_path,allow_pickle=True)
            pad_path = os.path.join(patient_id,'affine/padding_coordinate_conversion.npy')
            pad_v = np.load(pad_path,allow_pickle=True)

            # extract all parameters
            [Q, axis, angle, t_o, t_o_n, x_d, x_n, y_d, y_n, z_d, z_n, scale, t_c, t_c_n, img_center] = [M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8],M[9],M[10],M[11],M[12],M[13],M[14]]
            # center after padding
            image_center = img_center + pad_v
            mpr_center = img_center + t_c + pad_v

            # If *training*, we want to augment the data.
            # If *testing*, we do not.
            if self.augment:
                x, label,translation,rotation,scale,transform_matrix = self.image_data_generator.random_transform(x.astype("float32"), label.astype("float32"))
               
                #translation vector change
                t_c_n = dv.tf.change_of_translation_vector_after_augment(image_center, mpr_center ,transform_matrix,adapt_size)
               
                # direction vector change
                xx, x_len, x_n = dv.tf.change_of_direction_vector_after_augment(x_d,rotation,scale)
                yy, y_len, y_n = dv.tf.change_of_direction_vector_after_augment(y_d,rotation,scale)
                #zz, z_len, z_n = dv.tf.change_of_direction_vector_after_augment(z_d,rotation,scale)
                
                

            # Normalize the *individual* images to zero mean and unit std
            if self.normalize:
                batch_x[i] = dv.normalize_image(x)
            else:
                batch_x[i] = x

            batch_y1[i] = label
            batch_y2[i] = t_c_n
            batch_y3[i] = x_n
            batch_y4[i] = y_n
            
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
                self.image_data_generator.output_layer_names, [batch_y1,batch_y2,batch_y3,batch_y4]
            )
        }
        

        return (inputs, outputs)
