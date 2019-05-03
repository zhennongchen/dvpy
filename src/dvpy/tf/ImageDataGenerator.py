"""
Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...

For image segmentation problem data augmentation.
Transform train img data and mask img data simultaneously and in the same fashion.
Omit flow from directory function.
"""

# System

# Third Party
import numpy as np
from keras import backend as K

# Internal
import dvpy as dv


class ImageDataGenerator(object):
    """Generate minibatches with
    real-time data augmentation.
    Assume X is train img, Y is train label (same size as X with only 0 and 255 for values)
    # Arguments
        rotation_range: degrees (0 to 180). To X and Y
        shift_range: fraction of total height. To X and Y
        shear_range: shear intensity (shear angle in radians). To X and Y
        scale_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range. To X and Y
        channel_shift_range: shift range for each channels. Only to X
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'. For Y, always fill with constant 0
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally. To X and Y
        vertical_flip: whether to randomly flip images vertically. To X and Y
    """

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
        normalize=False,
    ):
        return dv.tf.NumpyArrayIterator(
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
            normalize=normalize,
        )

    def random_transform(self, x, y):

        translation,rotation,scale,transform_matrix_raw= dv.generate_random_transform(
            self.augmentation_params, x.shape[:-1]
        )
       
        transform_matrix = dv.transform_full_matrix_offset_center(
            transform_matrix_raw, x.shape[:-1]
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

    def zc_random_transform(self,x):
        translation,rotation,scale,matrix_raw=dv.zc_random(self.augmentation_params,x.shape[:-1])
        matrix_final=dv.transform_full_matrix_offset_center(matrix_raw,x.shape[:-1])
        print('channel_index: ',self.augmentation_params.img_channel_index,
        '\nfill_mode: ',self.augmentation_params.fill_mode,
            '\ncval: ',self.augmentation_params.cval)
        x_aug = dv.apply_affine_transform_channelwise(x,matrix_final[:-1, :], 
            channel_index=self.augmentation_params.img_channel_index,
            fill_mode=self.augmentation_params.fill_mode,
            cval=self.augmentation_params.cval)
        return x_aug,translation,rotation,scale,matrix_raw,matrix_final
