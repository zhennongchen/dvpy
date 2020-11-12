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
        slice_num = None,
        view = None,
        relabel_LVOT = None,
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
        self.slice_num = slice_num
        self.view = view
        self.relabel_LVOT = relabel_LVOT
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.shape = shape
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.augment = augment
        self.normalize = normalize
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, slice_num, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            # index_array is a randomly shuffled list of cases for this batch
            index_array, current_index, current_batch_size = next(self.index_generator)
            print('In numpy Array Iterator, index_array, current_index and current batch size = ',index_array, current_index, current_batch_size)
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
    

        # load the volume data
        volume_num = index_array[0][0]
        # image
        x = self.X[volume_num]
        print('The volume is ',x, volume_num)
        if self.input_adapter is not None:
            x = self.input_adapter(x)
            adapt_size = x.shape
            print('adapted volume has dimension, ',adapt_size)
        if self.normalize == 1:
            x = dv.normalize_image(x)

        # segmentation
        label = self.y[volume_num]
        print('The seg is ',label, volume_num)
        if self.output_adapter is not None:
            # ...and convert the path to a one-hot encoded image.
            label = self.output_adapter(label,self.relabel_LVOT)
            print('adapted segmentation has dimension ',label.shape)

        # load slice
        for i, j in enumerate(index_array):
            assert j[0] == volume_num
            print('slice num is ', j[1])
            image = x[:,:,j[1],:]   # !!!!
            seg = label[:,:,j[1],:]
            # If *training*, we want to augment the data.
            # If *testing*, we do not.
            if self.augment:
                image, seg,_,_,_,_ = self.image_data_generator.random_transform(image.astype("float32"), seg.astype("float32"))
                
            batch_x[i] = image
            batch_y1[i] = seg
            
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
