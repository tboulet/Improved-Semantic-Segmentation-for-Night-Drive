
import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D, 
    Conv2DTranspose, 
    Dropout,
    LeakyReLU, 
    UpSampling2D, 
)
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


### Mobilenet V2 ###

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


def UNet_MobileNetV2(image_size, num_classes) -> Model:

    encoder = MobileNetV2(
        input_shape=(image_size,image_size,3), 
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

    input = Input(shape=(image_size,image_size,3))
    bridges = down(input)
    x = bridges[-1]
    bridges = reversed(bridges[:-1])

    filters_up = [512, 256, 128, 64]
    i = 0
    for bridge in bridges:
        x = UpSampling2D()(x)
        x = Conv2D(filters_up[i], 3, padding="SAME")(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.5)(x)
        b = Conv2D(filters_up[i], 1, padding="SAME")(bridge)
        x = tf.concat([b, x], axis=-1)
        i += 1
    x = UpSampling2D()(x)
    x = Conv2D(num_classes, 1, activation="softmax")(x)

    return Model(input, x)


### DeepLabv3 ###

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3(image_size, num_classes) -> Model:
    model_input = Input(shape=(image_size, image_size, 3))
    resnet50 = ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='softmax')(x)
    return Model(inputs=model_input, outputs=model_output)


def make_model(config: dict) -> Model:
    """Gets a model from the run config"""
    model_name = config['model_name']
    image_size = config['image_size']
    num_classes = config['num_classes']

    if model_name == 'deeplabv3':
        return DeeplabV3(
            image_size=image_size, 
            num_classes=num_classes
        )
    elif model_name == 'unetmobilenetv2':
        return UNet_MobileNetV2(
            image_size=image_size, 
            num_classes=num_classes
        )
    else:
        raise NotImplementedError(model_name + " is not implemented !")


def get_encoder_unetmobilenetv2(model: Model):
    return Model(inputs=model.input, outputs=model.get_layer('conv2d').output)


def get_encoder_deeplabv3(model: Model):
    return Model(inputs=model.input, outputs=model.get_layer('conv2d').output)

    