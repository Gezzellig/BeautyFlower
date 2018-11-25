"""Functions relating to the structure of the network.
"""

from keras.layers import Input, BatchNormalization, Activation, Add
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam

class BeautyFlower:
    """The `BeautyFlower` GAN class.

    Parameters
    ------------
    n_residual_blocks: `int` (default=6)
        Number of blocks of intermidiary layers in the generator and discriminator.
    learning_rate: `float` (default=0.001)
        Learning rate for the Adam optimizer.
    """

    def __init__(self, n_residual_blocks=6, learning_rate=0.001):

        # Input shape
        self.channels   = 3

        # Low-resolution dimensions
        self.lr_height  = 16
        self.lr_width   = 16
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)

        # High-resolution dimensions
        self.hr_height  = 32
        self.hr_width   = 32
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Define amounts of initial filters in the generator and discriminator
        self.gf = 16
        self.df = 16

        # Number of blocks of layers to be added in the middle of the sequence.
        # The structures of the blocks are defined in the buildGenerator and buildDiscriminator functions.
        self.n_residual_blocks = n_residual_blocks

        # The learning rate for the Adam optimizer used in the subnets.
        self.learning_rate = learning_rate

        # Make the discriminator and generator
        self.generator      = self.buildGenerator()
        self.discriminator  = self.buildDiscriminator()

        self.generator.compile(loss='binary_crossentropy',
                        loss_weights=[1e-3],
                        optimizer=Adam(self.learning_rate))

    def buildGenerator(self):
        """Builds the generator of the network using building blocks of layers
        """

        # Input layer with the shape of the low-res images
        inputLayer = Input(shape=self.lr_shape)

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
            u = Conv2D(64, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # First block after the input layer
        c1 = Conv2D(16, kernel_size=9, strides=1, padding='same')(inputLayer)
        c1 = Activation('relu')(c1)

        # Make a number of residual blocks each connected to the previous layer
        r = generatorBlock(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = generatorBlock(r, self.gf)

        # Post-residual block
        c2 = Conv2D(16, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling layer
        upsamplingLayer = upsamplingBlock(c2)

        # Obtain high-resolution image
        generatedOutput = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(upsamplingLayer)

        return Model(inputLayer, generatedOutput)

    def buildDiscriminator(self):
        """Builds the discriminator of the network using building blocks of layers
        """
        pass

    def trainGenerator(self, lowResData, highResData):
        # For now set this equal to the length of the data we give it
        # TODO implement proper batch size and data loading

        # Train the generators
        g_loss = self.generator.train_on_batch(lowResData, highResData)

        return g_loss
    
    def generateImage(self, image):
        generatedHighRes = self.generator.predict(image)
        return generatedHighRes