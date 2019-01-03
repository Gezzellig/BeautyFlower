"""Functions relating to the structure of the network.
"""
import numpy as np
from keras.layers import Input, BatchNormalization, Activation, Add, concatenate, LeakyReLU, Dropout, Dense, Flatten
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

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
        self.lr_height  = 48
        self.lr_width   = 48
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)

        # High-resolution dimensions
        self.hr_height  = 96
        self.hr_width   = 96
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
        self.discriminator  = self.buildDiscriminator()
        self.generator      = self.buildGenerator()

        self.discriminator.compile(loss='binary_crossentropy',
                        loss_weights=[1e-3],
                        optimizer=Adam(self.learning_rate))

        self.generator.compile(loss='binary_crossentropy',
                        loss_weights=[1e-3],
                        optimizer=Adam(self.learning_rate))


    def buildGenerator(self):
        """Builds the generator of the network using building blocks of layers
        """

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
        initial_filters = self.gf

        current_filter = initial_filters * upsample_scale

        # Input layer with the shape of the low-res images
        inputLayer = Input(shape=self.lr_shape)

        # Upsample the input by a factor of 2
        u1 = UpSampling2D((upsample_scale, upsample_scale))(inputLayer)

        # First block after the input layer
        c1 = Conv2D(current_filter, kernel_size=9, strides=1, padding='same')(u1)

        d1 = denseBlock(c1, current_filter, 3)

        # Obtain high-resolution image
        # generatedOutput = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(d1)
        generatedOutput = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='linear')(d1)

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
        current_filters = self.df

        # Initial layers
        c1 = Conv2D(current_filters, kernel_size=3, strides=1, padding='same')(inputLayer)
        l1 = LeakyReLU(alpha=0.2)(c1)
        d1 = Dropout(0.25)(l1)

        # Denseblocks
        db1 = denseBlock(d1, current_filters, amount_layers=1)

        # Get single activation
        flat1  = Flatten()(db1)
        dense1 = Dense(1, activation="sigmoid")(flat1)

        return Model(inputLayer, dense1)

    def train(self, lr_images, hr_images, batch_size=100):
        ########################
        # TRAIN DISCRIMINATOR
        ########################
        
        # List of 1's as the positive feedback for the real images
        postive_feedback  = np.ones(batch_size)

        # List of 0's as the negative feedback for the fake images
        negative_feedback = np.zeros(batch_size)

        # Rescale the image pixel values from -1 to 1
        lr_images = (lr_images.astype(np.float32) - 127.5) / 127.5
        hr_images = (hr_images.astype(np.float32) - 127.5) / 127.5

        # Get random batch from the training set (select [batch_size] ints from range 0 - lr_images_length)
        idx = np.random.randint(0, lr_images.shape[0], batch_size)
        
        # Obtain selected images from batch indices
        imgs = lr_images[idx]

        # Generate a new random image based on the low res images selected.
        latent_fake = self.generator.predict(imgs)
        # print(((lr_images+1)*127.5).astype(np.uint8)[0])
        # plt.figure()
        # plt.imshow(((lr_images+1)*127.5).astype(np.uint8)[0])
        # plt.show()
        latent_real = hr_images[idx]

        # Start training on real images, and then on fake images and calculate the loss
        d_loss_real = self.discriminator.train_on_batch(latent_real, postive_feedback)
        d_loss_fake = self.discriminator.train_on_batch(latent_fake, negative_feedback)

        # Combine the losses from the real and the fake
        d_loss_average = 0.5 * np.add(d_loss_real, d_loss_fake)

        ########################
        # TRAIN GENERATOR
        ########################

        # Train generator on same random indices as the discriminator
        g_loss = self.generator.train_on_batch(lr_images[idx], hr_images[idx])


    def trainGenerator(self, epochs, lowResData, highResData):
        # For now set this equal to the length of the data we give it
        # TODO implement proper batch size and data loading

        # Train the generators

        g_loss = self.generator.train_on_batch(lowResData, highResData)

        return g_loss
    
    def generateImage(self, image):
        generatedHighRes = self.generator.predict(image)
        return generatedHighRes