
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D, 
    Conv2DTranspose, 
    DepthwiseConv2D,
    Dropout,
    LeakyReLU, 
    UpSampling2D, 
    ZeroPadding2D
)
from tensorflow_addons.layers import InstanceNormalization


class Block1(Model):

    def __init__(
        self, 
        input_channels: int = 16,
        output_channels: int = 16,
        expansion_factor: int = 6,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.expansion_factor = expansion_factor

        self.use_batch_norm = use_batch_norm

        mid_channels = self.input_channels * self.expansion_factor 
        self.conv1 = Conv2D(mid_channels, (1,1), (1,1), padding='SAME')
        self.dconv1 = DepthwiseConv2D((3,3), (1,1), padding='SAME')
        self.conv2 = Conv2D(output_channels, (1,1), (1,1), padding='SAME')
        self.conv3 = Conv2D(output_channels, (1,1), (1,1), padding='SAME')

    def __call__(self, input_tensor: tf.Tensor, training=True) -> tf.Tensor:
       
        m = self.conv1(input_tensor)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
        m = tf.nn.relu6(m)
        m = self.dconv1(m)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
        m = tf.nn.relu6(m)
        m = self.conv2 (m)
        n = self.conv3(input_tensor)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
            n = tf.nn.batch_normalization(n)

        return m + n

class Block2(Model):

    def __init__(
        self, 
        input_channels: int = 16,
        output_channels: int = 16,
        expansion_factor: int = 6,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.expansion_factor = expansion_factor

        self.use_batch_norm = use_batch_norm

        mid_channels = self.input_channels * self.expansion_factor 
        self.conv1 = Conv2D(mid_channels, (1,1), (1,1), padding='SAME')
        self.dconv1 = DepthwiseConv2D((3,3), (2,2), padding='VALID')
        self.conv2 = Conv2D(output_channels, (1,1), (1,1), padding='SAME')
    
    def __call__(self, input_tensor: tf.Tensor, training=True) -> tf.Tensor:

        m = self.conv1(input_tensor)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
        m = tf.nn.relu6(m)
        m = self.dconv1(m)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)
        m = tf.nn.relu6(m)
        m = self.conv2(m)
        if self.use_batch_norm:
            m = tf.nn.batch_normalization(m)

        return m

class MobileUNET_XS(Model):

    def __init__(
        self, 
        num_classes: int = 10,
        use_batch_norm: bool = False
    ) -> None:
        super().__init__()
        
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
        if m5.shape != m7.shape:
            m7 = tf.image.resize_with_crop_or_pad(m7, m5.shape[1], m5.shape[2])
        m7 = tf.concat([m5, m7], axis=-1)

        m8 = Conv2DTranspose(64, (3,3), (2,2), padding='SAME')(m7)
        if ubn:
            m8 = tf.nn.batch_normalization(m8)
        m8 = tf.nn.relu6(m8)
        if m4.shape != m8.shape:
            m8 = tf.image.resize_with_crop_or_pad(m8, m4.shape[1], m4.shape[2])
        m8 = tf.concat([m4, m8], axis=-1)

        m9 = Conv2DTranspose(32, (3,3), (2,2), padding='SAME')(m8)
        if ubn:
            m9 = tf.nn.batch_normalization(m9)
        m9 = tf.nn.relu6(m9)
        if m3.shape != m9.shape:
            m9 = tf.image.resize_with_crop_or_pad(m9, m3.shape[1], m3.shape[2])
        m9 = tf.concat([m3, m9], axis=-1)

        m10 = Conv2DTranspose(24, (3,3), (2,2), padding='SAME')(m9)
        if ubn:
            m10 = tf.nn.batch_normalization(m10)
        m10 = tf.nn.relu(m10)
        if m2.shape != m10.shape:
            m10 = tf.image.resize_with_crop_or_pad(m10, m2.shape[1], m2.shape[2])
        m10 = tf.concat([m2, m10], axis=-1)

        m11 = Conv2DTranspose(16, (3,3), (2,2), padding='SAME')(m10)
        if ubn:
            m11 = tf.nn.batch_normalization(m11)
        m11 = tf.nn.relu6(m11)
        if m1.shape != m11.shape:
            m11 = tf.image.resize_with_crop_or_pad(m11, m1.shape[1], m1.shape[2])
        m11 = tf.concat([m1, m11], axis=-1)

        m12 = Conv2DTranspose(self.num_classes, (3,3), (2,2), padding='SAME')(m11)

        return m12

class MobileUNET_S(Model):

    def __init__(
        self, 
        num_classes: int = 10,
        use_batch_norm: bool = False
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes

        self.use_batch_norm = ubn = use_batch_norm

        self.conv1 = Conv2D(32, (3,3), (1,1), padding='SAME')
        self.conv2 = Conv2D(32, (3,3), (2,2), padding='SAME')

        self.block1_1 = Block1(32, 16, 1, ubn)

        self.block2_1 = Block2(16, 24, 6, ubn)
        self.block2_2 = Block1(24, 24, 6, ubn)
        self.block2_3 = Block1(24, 24, 6, ubn)

        self.block3_1 = Block2(24, 32, 6, ubn)
        self.block3_2 = Block1(32, 32, 6, ubn)
        self.block3_3 = Block1(32, 32, 6, ubn)

        self.block4_1 = Block2(32, 64, 6, ubn)
        self.block4_2 = Block1(64, 64, 6, ubn)
        self.block4_3 = Block1(64, 64, 6, ubn)

        self.block5_1 = Block2(64, 128, 6, ubn)
        self.block5_2 = Block1(128, 128, 6, ubn)
        self.block5_3 = Block1(128, 128, 6, ubn)

        self.block6_1 = Block2(128, 256, 6, ubn)
        self.block6_2 = Block1(256, 256, 6, ubn)
        self.block6_3 = Block1(256, 256, 6, ubn)

        self.convt1 = Conv2DTranspose(128, (3,3), (2,2), padding='SAME')
        self.conv3 = Conv2D(128, (3,3), (1,1), padding='SAME')

        self.convt2 = Conv2DTranspose(64, (3,3), (2,2), padding='SAME')
        self.conv4 = Conv2D(64, (3,3), (1,1), padding='SAME')

        self.convt3 = Conv2DTranspose(32, (3,3), (2,2), padding='SAME')
        self.conv5 = Conv2D(32, (3,3), (1,1), padding='SAME')

        self.convt4 = Conv2DTranspose(24, (3,3), (2,2), padding='SAME')
        self.conv6 = Conv2D(24, (3,3), (1,1), padding='SAME')

        self.convt5 = Conv2DTranspose(16, (3,3), (2,2), padding='SAME')
        self.conv7 = Conv2D(16, (3,3), (1,1), padding='SAME')

        self.convt6 = Conv2DTranspose(32, (3,3), (2,2), padding='SAME')
        self.conv8 = Conv2D(self.num_classes, (3,3), (1,1), padding='SAME', kernel_initializer=tf.keras.initializers.Zeros())
    
    def __call__(self, input_tensor: tf.Tensor, training=True) -> tf.Tensor:

        ubn = self.use_batch_norm
        
        ### First Convolution ##
        m0 = self.conv1(input_tensor)
        m = self.conv2(m0)

        ### Encoder ###
        m1 = self.block1_1(m)

        m2 = self.block2_1(m1)
        m2 = self.block2_2(m2)
        m2 = self.block2_3(m2)

        m3 = self.block3_1(m2)
        m3 = self.block3_2(m3)
        m3 = self.block3_3(m3)

        m4 = self.block4_1(m3)
        m4 = self.block4_2(m4)
        m4 = self.block4_3(m4)

        m5 = self.block5_1(m4)
        m5 = self.block5_2(m5)
        m5 = self.block5_3(m5)

        m6 = self.block6_1(m5)
        m6 = self.block6_2(m6)
        m6 = self.block6_3(m6)

        ### Decoder ###
        m7 = self.convt1(m6)
        if ubn:
            m7 = tf.nn.batch_normalization(m7)
        m7 = tf.nn.relu6(m7)
        if m5.shape != m7.shape:
            m7 = ZeroPadding2D()(m7)
            # m5 = m5[:,:-1,:-1,:]
            pass
        # print(m5.shape, m7.shape)
        m7 = tf.concat([m5, m7], axis=-1)
        m7 = self.conv3(m7)

        m8 = self.convt2(m7)
        if ubn:
            m8 = tf.nn.batch_normalization(m8)
        m8 = tf.nn.relu6(m8)
        if m4.shape != m8.shape:
            m8 = ZeroPadding2D()(m8)
            m8 = m8[:,:-1,:-1,:]
            pass
        # print(m4.shape, m8.shape)
        m8 = tf.concat([m4, m8], axis=-1)
        m8 = self.conv4(m8)

        m9 = self.convt3(m8)
        if ubn:
            m9 = tf.nn.batch_normalization(m9)
        m9 = tf.nn.relu6(m9)
        if m3.shape != m9.shape:
            m9 = ZeroPadding2D()(m9)
            m9 = m9[:,:-1,:-1,:]
            pass
        # print(m3.shape, m9.shape)
        m9 = tf.concat([m3, m9], axis=-1)
        m9 = self.conv5(m9)

        m10 = self.convt4(m9)
        if ubn:
            m10 = tf.nn.batch_normalization(m10)
        m10 = tf.nn.relu(m10)
        if m2.shape != m10.shape:
            m10 = ZeroPadding2D()(m10)
            m10 = m10[:,:-1,:-1,:]
            pass
        # print(m2.shape, m10.shape)
        m10 = tf.concat([m2, m10], axis=-1)
        m10 = self.conv6(m10)
        m10 = tf.nn.relu6(m10)

        m11 = self.convt5(m10)
        if ubn:
            m11 = tf.nn.batch_normalization(m11)
        m11 = tf.nn.relu6(m11)
        if m1.shape != m11.shape:
            m11 = ZeroPadding2D()(m11)
            # m1 = m1[:,:-1,:-1,:]
            pass
        # print(m1.shape, m11.shape)
        m11 = tf.concat([m1, m11], axis=-1)
        m11 = self.conv7(m11)
        m11 = tf.nn.relu6(m11)

        m12 = self.convt6(m11)
        m12 = tf.nn.relu6(m12)
        # print(m0.shape, m12.shape)
        m12 = tf.concat([m0, m12], axis=-1)
        m12 = self.conv8(m12)
        
        return tf.nn.softmax(m12, axis=-1)

class PreTrainedMobileUnetv2(Model):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.encoder = MobileNetV2((224,224,3), include_top=False)

        self.convt1 = Conv2DTranspose(576, (3,3), (2,2), padding='SAME')
        self.conv1 = Conv2D(576, (3,3), (1,1), padding='SAME')

        self.convt2 = Conv2DTranspose(192, (3,3), (2,2), padding='SAME')
        self.conv2 = Conv2D(192, (3,3), (1,1), padding='SAME')

        self.convt3 = Conv2DTranspose(144, (3,3), (2,2), padding='SAME')
        self.conv3 = Conv2D(144, (3,3), (1,1), padding='SAME')

        self.convt4 = Conv2DTranspose(96, (3,3), (2,2), padding='SAME')
        self.conv4 = Conv2D(96, (3,3), (1,1), padding='SAME')

        self.convt5 = Conv2DTranspose(32, (3,3), (2,2), padding='SAME')
        self.conv5 = Conv2D(self.num_classes, (3,3), (1,1), padding='SAME', kernel_initializer=tf.keras.initializers.Zeros())

        self.bridge_layer_names = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu']
 

    def __call__(self, input_tensor: tf.Tensor, training=True) -> tf.Tensor:

        encoded = self.encoder(input_tensor, training=True)
        bridges = [self.encoder.get_layer(layer_name).output for layer_name in self.bridge_layer_names]

        up1 = self.convt1(encoded)
        up1 = tf.concat([up1, bridges[-1]], axis=-1)
        up1 = self.conv1(up1)

        up2 = self.convt2(up1)
        up2 = tf.concat([up2, bridges[-2]], axis=-1)
        up2 = self.conv2(up2)

        up3 = self.convt3(up2)
        up3 = tf.concat([up3, bridges[-3]], axis=-1)
        up3 = self.conv3(up3)

        up4 = self.convt4(up3)
        up4 = tf.concat([up4, bridges[-4]], axis=-1)
        up4 = self.conv4(up4)

        up5 = self.convt5(up4)
        up5 = self.conv5(up5)

        return up5


    def call(self, input_tensor, training=True):
        return self.__call__(input_tensor, training=True)


def u_net_pretrained(num_classes):

    encoder = MobileNetV2(
        input_shape=(224,224,3), 
        include_top=False,
        weights='imagenet'
    )
    bridge_layers = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu', 
        'block_13_expand_relu',
        'block_16_project',
    ]
    encoder_outputs = [encoder.get_layer(layer).output for layer in bridge_layers]
    down = Model(inputs=encoder.input, outputs=encoder_outputs)

    input = Input(shape=(224,224,3))
    bridges = down(input)
    x = bridges[-1]
    bridges = reversed(bridges[:-1])

    filters_up = [512, 256, 128, 64]
    i = 0
    for bridge in bridges:
        x = UpSampling2D()(x)
        x = Conv2D(filters_up[i], 3, padding="SAME")(x)
        x = InstanceNormalization()(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.5)(x)
        b = Conv2D(filters_up[i], 1, padding="SAME")(bridge)
        x = tf.concat([b, x], axis=-1)
        i += 1
    x = UpSampling2D()(x)
    x = Conv2D(num_classes, 1, activation="softmax")(x)
    # print(x)

    return Model(input, x)

