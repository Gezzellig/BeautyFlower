"""
Main program
- Loads the data
- Runs the network
"""

import network
import dataset
import random

# Load data
data = dataset.getData([2], show_info=True)

# Show a random example image
dataset.showImage(data, random.randint(0, len(data) - 1), seconds=2)