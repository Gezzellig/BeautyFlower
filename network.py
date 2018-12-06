"""Functions relating to the structure of the network.
"""

from keras.layers import Input, BatchNormalization, Activation, Add, concatenate, LeakyReLU, Dropout, Dense, Flatten
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

        def denseFactor(layer_input, filters):
            """Single layer in the dense blocks
            """
            layer = layer_input

            layer = BatchNormalization(momentum=0.8)(layer)
            layer = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer)
            layer = Activation('relu')(layer)

            return layer
        
        def denseBlock(layer_inputs, filters, amount_layers):
            """Block of dense factors with dense connections
            """
            concatenated_inputs = layer_inputs

            for _ in range(amount_layers):
                x = denseFactor(concatenated_inputs, filters)
                concatenated_inputs = concatenate([concatenated_inputs, x], axis=3)

            return concatenated_inputs

        # Scale variable upsampling
        upsample_scale  = 2
        initial_filters = self.lr_height

        current_filter = initial_filters * upsample_scale

        # Upsample the input by a factor of 2
        u1 = UpSampling2D((upsample_scale, upsample_scale))(inputLayer)

        # First block after the input layer
        c1 = Conv2D(current_filter, kernel_size=9, strides=1, padding='same')(u1)

        d1 = denseBlock(c1, current_filter, 3)

        # Obtain high-resolution image
        generatedOutput = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(d1)

        return Model(inputLayer, generatedOutput)

    def buildDiscriminator(self):
        """Builds the discriminator of the network using building blocks of layers
        """

        def denseFactor(layer_input, filters):
            """Single layer in the dense blocks
            """
            layer = layer_input

            layer = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer)
            layer = BatchNormalization(momentum=0.8)(layer)
            layer = LeakyReLU(alpha=0.2)(layer)
            layer = Dropout(0.25)(layer)

            return layer

        def denseBlock(layer_inputs, filters, amount_layers):
            """Block of dense factors with dense connections
            """
            concatenated_inputs = layer_inputs
            filter_multiplier   = 2

            for _ in range(amount_layers):
                x = denseFactor(concatenated_inputs, filters)
                concatenated_inputs = concatenate([concatenated_inputs, x], axis=3)
                filters = filters * filter_multiplier
            return concatenated_inputs

        # Input layer with the shape of the high-res images
        inputLayer = Input(shape=self.hr_shape)
        current_filters = self.hr_height

        # Initial layers
        c1 = Conv2D(current_filters, kernel_size=3, strides=1, padding='same')(inputLayer)
        l1 = LeakyReLU(alpha=0.2)(c1)
        d1 = Dropout(0.25)(l1)

        # Denseblocks
        db1 = denseBlock(d1, current_filters, amount_layers=3)

        # Get single activation
        flat1  = Flatten()(db1)
        dense1 = Dense(1, activation="sigmoid")(flat1)

        return Model(inputLayer, dense1)

    def trainGenerator(self, lowResData, highResData):
        # For now set this equal to the length of the data we give it
        # TODO implement proper batch size and data loading

        # Train the generators
        g_loss = self.generator.train_on_batch(lowResData, highResData)

        return g_loss
    
    def generateImage(self, image):
        generatedHighRes = self.generator.predict(image)
        return generatedHighRes