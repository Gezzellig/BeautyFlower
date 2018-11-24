import network
import dataset
import random

"""
Main program
- Loads the data
- Runs the network
"""

# Load data
data = dataset.getData([2], show_info=True)

# Show a random example image
dataset.showImage(data, random.randint(0, 2499))