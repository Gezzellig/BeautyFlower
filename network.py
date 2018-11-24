"""Functions relating to the structure of the network.
"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

class BeautyFlower:
    def __init__(self):
        # Input shape
        self.channels   = 3

        # Low-resolution dimensions
        self.lr_height  = 16
        self.lr_width   = 16
        self.lr_shape = (self.hr_height, self.hr_width, self.channels)

        # High-resolution dimensions
        self.hr_height  = 32
        self.hr_width   = 32
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Make the discriminator and generator
        self.generator      = self.buildGenerator()
        self.discriminator  = self.buildDiscriminator()

        def buildGenerator(self):
            """Builds the generator of the network using building blocks of layers
            """

            def generatorBlock(layer_input, filters):
                """Residual block for the generator based on the SRGAN blocks
                """
                d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
                d = Activation('relu')(d)
                d = BatchNormalization(momentum=0.8)(d)
                d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
                d = BatchNormalization(momentum=0.8)(d)
                d = Add()([d, layer_input])
                return d

            def upsamplingBlock(layer_input):
                """Upsampling layer based on SRGAN blockss"""
                u = UpSampling2D(size=2)(layer_input)
                u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
                u = Activation('relu')(u)
                return u

            # Input layer
            inputLayer = Input(shape=self.lr_shape)

            # Pre-residual block
            c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(inputLayer)
            c1 = Activation('relu')(c1)
            pass

        def buildDiscriminator(self):
            """Builds the discriminator of the network using building blocks of layers
            """
            pass