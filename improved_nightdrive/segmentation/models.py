
from typing import Tuple, Optional

import tensorflow as tf

import tensorflow.keras as keras
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D

class Block1(Model):

    def __init__(
        self, 
        output_channels: int = 16,
        expansion_factor: int = 6,
        use_batch_norm: bool = False,
    ) -> None:

        self.output_channels = output_channels
        self.expansion_factor = expansion_factor

        self.use_batch_norm = use_batch_norm

    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:
        
        mid_channels = input_tensor.shape[-1] * self.expansion_factor
        output_channels = self.output_channels

        m = Conv2D(mid_channels, (1,1), (1,1), padding='SAME')(input_tensor)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
        m = tf.nn.relu6(m)
        m = DepthwiseConv2D((3,3), (1,1), padding='SAME' )(m)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
        m = tf.nn.relu6(m)
        m = Conv2D(output_channels, (1,1), (1,1), padding='SAME')(m)
        n = Conv2D(output_channels, (1,1), (1,1), padding='SAME')(input_tensor)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
            n = tf.nn.batch_normalization(n)

        return m + n

class Block2(Model):

    def __init__(
        self, 
        output_channels: int = 16,
        expansion_factor: int = 6,
        use_batch_norm: bool = False,
    ) -> None:

        self.output_channels = output_channels
        self.expansion_factor = expansion_factor

        self.use_batch_norm = use_batch_norm
    
    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:

        mid_channels = input_tensor.shape[-1] * self.expansion_factor
        output_channels = self.output_channels

        m = Conv2D(mid_channels, (1,1), (1,1), padding='SAME')(input_tensor)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
        m = tf.nn.relu6(m)
        m = DepthwiseConv2D((3,3), (2,2), padding='SAME')(m)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
        m = tf.nn.relu6(m)
        m = Conv2D(output_channels, (1,1), (1,1), padding='SAME')(m)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)

        return m

class MobileUNET_XS(Model):

    def __init__(
        self, 
        num_classes: int = 10,
        use_batch_norm: bool = False
    ) -> None:
        
        self.num_classes = num_classes

        self.use_batch_norm = use_batch_norm
    
    def __call__(self, input_tensor: tf.Tensor) -> tf.Tensor:

        ubn = self.use_batch_norm
        
        ### First Convolution ##
        m = Conv2D(32, (3,3), (2,2), padding='SAME')(input_tensor)

        ### Encoder ###
        m1 = Block1(16, 1, ubn)(m)

        m2 = Block2(24, 6, ubn)(m1)
        m2 = Block1(24, 6, ubn)(m2)

        m3 = Block2(32, 6, ubn)(m2)
        m3 = Block1(32, 6, ubn)(m3)

        m4 = Block2(64, 6, ubn)(m3)
        m4 = Block1(64, 6, ubn)(m4)

        m5 = Block2(64, 6, ubn)(m4)
        m5 = Block1(64, 6, ubn)(m5)

        m6 = Block2(32, 6, ubn)(m5)
        m6 = Block1(32, 6, ubn)(m6)

        ### Decoder ###
        m7 = Conv2DTranspose(64, (3,3), (2,2), padding='SAME')(m6)
        if ubn:
            m7 = tf.nn.batch_normalization(m7)
        m7 = tf.nn.relu6(m7)
        m7 = tf.concat([m5, m7], axis=-1)

        m8 = Conv2DTranspose(64, (3,3), (2,2), padding='SAME')(m7)
        if ubn:
            m8 = tf.nn.batch_normalization(m8)
        m8 = tf.nn.relu6(m8)
        m8 = tf.concat([m4, m8], axis=-1)

        m9 = Conv2DTranspose(32, (3,3), (2,2), padding='SAME')(m8)
        if ubn:
            m9 = tf.nn.batch_normalization(m9)
        m9 = tf.nn.relu6(m9)
        m9 = tf.concat([m3, m9], axis=-1)

        m10 = Conv2DTranspose(24, (3,3), (2,2), padding='SAME')(m9)
        if ubn:
            m10 = tf.nn.batch_normalization(m10)
        m10 = tf.nn.relu(m10)
        m10 = tf.concat([m2, m10], axis=-1)

        m11 = Conv2DTranspose(16, (3,3), (2,2), padding='SAME')(m10)
        if ubn:
            m11 = tf.nn.batch_normalization(m11)
        m11 = tf.nn.relu6(m11)
        m11 = tf.concat([m1, m11], axis=-1)

        m12 = Conv2DTranspose(self.num_classes, (3,3), (2,2), padding='SAME')(m11)

        return m12

