import preload
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Activation

original, smaller, bicubic = preload.load_preload_images_batch("data/learnset", 10, 0)





def generator():
	model = Sequential([
    	Dense(32, input_shape=(784,)),
    	Activation('relu'),
    	Dense(10),
    	Activation('softmax'),
	])