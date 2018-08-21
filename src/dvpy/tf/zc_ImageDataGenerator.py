# System

# Third Party
import numpy as np
from keras import backend as K

# Internal
import dvpy as dv


class zc_ImageDataGenerator(object):
    
    def __init__(
        self,
        image_dimension,
        input_layer_names,
        output_layer_names,
        translation_range=0.,
        rotation_range=0.,
        scale_range=0.,
        flip=False,
        fill_mode="constant",
        cval=0.,
    ):

        self.input_layer_names = input_layer_names
        self.output_layer_names = output_layer_names
        self.augmentation_params = dv.AugmentationParameters(
            image_dimension,
            translation_range=translation_range,
            rotation_range=rotation_range,
            scale_range=scale_range,
            flip=flip,
            fill_mode=fill_mode,
            cval=cval,
        )

        if K.image_dim_ordering() != "tf":
            raise Exception("Only tensorflow backend is supported.")

    def flow(
        self,
        X,
        y=None,
        batch_size=32,
        shuffle=True,
        seed=None,
        input_adapter=None,
        output_adapter=None,
        shape=None,
        input_channels=None,
        output_channels=None,
        augment=False,
    ):
        return dv.tf.zc_NumpyArrayIterator(
            X,
            y,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            input_adapter=input_adapter,
            output_adapter=output_adapter,
            shape=shape,
            input_channels=input_channels,
            output_channels=output_channels,
            augment=augment,
        )

    def random_transform(self, x, y):

        translation,rotation,scale,transform_matrix = dv.zc_generate_random_transform(
            self.augmentation_params, x.shape[:-1]
        )
        transform_matrix = dv.transform_full_matrix_offset_center(
            transform_matrix, x.shape[:-1]
        )
        x = dv.apply_affine_transform_channelwise(
            x,
            transform_matrix[:-1, :],
            channel_index=self.augmentation_params.img_channel_index,
            fill_mode=self.augmentation_params.fill_mode,
            cval=self.augmentation_params.cval,
        )

        # For y, mask data, fill mode constant, cval = 0
        y = dv.apply_affine_transform_channelwise(
            y,
            transform_matrix[:-1, :],
            channel_index=self.augmentation_params.img_channel_index,
            fill_mode=self.augmentation_params.fill_mode,
            cval=self.augmentation_params.cval,
        )

        return x, y,translation,rotation,scale,transform_matrix
