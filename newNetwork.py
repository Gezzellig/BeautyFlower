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

    def build_generator(self):
        # Input layer with the shape of the low-res images
        input_layer = Input(shape=self.lr_shape)