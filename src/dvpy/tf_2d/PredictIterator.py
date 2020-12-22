# System

# Third Party
import numpy as np
from keras import backend as K
import sympy as sym

# Internal
import dvpy as dv
from . import IteratorBase
import os

class PredictIterator(IteratorBase):
    def __init__(
        self,
        X,
        y,
        image_data_generator,
        slice_num = None,
        batch_size = None,
        patients_in_one_batch = None,
        relabel_LVOT = None,
        shuffle=None,
        seed=None,
        input_adapter=None,
        output_adapter=None,
        shape=None,
        input_channels=None,
        output_channels=None,
        augment=False,
        normalize=False,
        adapted_already = None,
    ):

        if K.image_dim_ordering() != "tf":
            raise Exception("Only tensorflow backend is supported.")

   
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.slice_num = slice_num
        self.batch_size = batch_size
        self.patients_in_one_batch = patients_in_one_batch
        self.relabel_LVOT = relabel_LVOT
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.shape = shape
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.augment = augment
        self.normalize = normalize
        self.adapted_already = adapted_already
        super(PredictIterator, self).__init__(X.shape[0], slice_num, batch_size, patients_in_one_batch, shuffle, seed)

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
    

    # load slice
        if self.shuffle == True:
            index_array = index_array.tolist()
            index_array.sort()
      
        volumes_already_load = []
        for i, j in enumerate(index_array):
            case = j[0] #######
            if case not in volumes_already_load:
                volumes_already_load.append(case)
                # load volume + seg:
                x = self.X[case]
                
                if self.adapted_already == 0:
                    if self.input_adapter is not None:
                        x = self.input_adapter(x)
                        adapt_size = x.shape
                    if self.normalize == 1:
                        x = dv.normalize_image(x)
                elif self.adapted_already == 1:
                    x = np.load(x,allow_pickle = True)
                   
                    
                else:
                    raise ValueError('wrong definition of adapted_already')
           
            image = x[:,:,j[1],:]   # !!!!
            batch_x[i] = image
            
            
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
        