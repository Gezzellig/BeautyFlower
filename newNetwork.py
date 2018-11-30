"""Functions relating to the structure of the network.
"""

from keras.layers import Input, BatchNormalization, Activation, Add, concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam

class superGAN:
    def __init__(self):
        # Input shape
        self.channels   = 3

        # Low-resolution dimensions
        self.lr_height  = 16
        self.lr_width   = 16
        self.lr_shape   = (self.lr_height, self.lr_width, self.channels)

        # High-resolution dimensions
        self.hr_height  = 32
        self.hr_width   = 32
        self.hr_shape   = (self.hr_height, self.hr_width, self.channels)

        # Define amounts of initial filters in the generator and discriminator
        self.gf = 16
        self.df = 16

        # Build the networks
        self.build_generator()

    def dcu_factor (self, layer_input, filters):
        """Single layer in the dense blocks
        """
        layer = layer_input

        # Batch Normalization:
        # Normalize the activations of the previous layer at each batch.
        # I.e. applies a transformation that maintains the mean activation close 
        # to 0 and the activation standard deviation close to 1.
        layer = BatchNormalization(momentum=0.8)(layer)

        # ReLu:
        # Output shape is the same as the input shape.
        # With default values, it returns element-wise max(x, 0).
        layer = Activation('relu')(layer)

        # Convolution layer:
        # Output shape is the same as the shape of the original input with arguments:
        # strides = 1, and padding = 'same'
        layer = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer)

        return layer

    def build_generator(self):
        # Input layer with the shape of the low-res images
        input_layer = Input(shape=self.lr_shape)


# Notes: 
# Padding: "same" results in padding the input such that the output has the same length as the original input.