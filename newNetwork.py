"""Functions relating to the structure of the network.
"""

from keras.layers import Input, BatchNormalization, Activation, Add, concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam

class superGAN:
