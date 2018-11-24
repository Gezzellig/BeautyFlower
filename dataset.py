import matplotlib.pyplot as plt
import numpy as np

"""
Functions relating to handling the data.
Makes extracting data from CIFAR dataset a little easier.
Expects CIFAR-100 dataset in 'data/' folder

Example usage for extracting only aquatic_mammals category pictures: 
    data = getData([0], show_info=True)
"""

def unpickle(file):
    """
    Decompression function required to load the CIFAR data.
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def printLabelNames(indices):
    """
    Prints the label names for coarse label indices
    Params:
        indices: List of indices corresponding to CIFAR categories.
    """
    meta = unpickle("data/meta")
    label_names = meta[b'coarse_label_names']
    labels = [label_names[index] for index in indices]
    print ("Selected categories: " + str(labels))

def getData(target_labels, show_info=False):
    """
    Gets data from folder 'data/' which should contain Cifar-100 database.
    Params:
        target_labels: list of course label indices included in the final dataset
        show_info: Will print some info about the amount of images, and the included categories.
    """

    # Decompress the data
    data = unpickle("data/train")

    # Extract data and labels from dataset
    x_train = data[b'data']
    y_train = data[b'coarse_labels']

    # Select indices of labels that are in the target labels
    selected_indices_training = [idx for idx, label in enumerate(y_train) if label in target_labels]

    # Select only images from target categories to input
    filtered_data = x_train[selected_indices_training]
    
    if show_info:
        # Output selected categories
        printLabelNames(target_labels)
    
        # Images in selected data
        print ("Images in selected categories: " + str(len(filtered_data)))
    
    return filtered_data