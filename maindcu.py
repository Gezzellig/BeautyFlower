import network

gan = network.BeautyFlower()
print("generator summary:")
gan.generator.summary()
input()
print("discriminator summary:")
gan.discriminator.summary()
