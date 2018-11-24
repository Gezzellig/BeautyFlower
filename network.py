import matplotlib.pyplot as plt
import numpy as np
import keras

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def printLabelNames(indices):
    meta = unpickle("data/meta")
    label_names = meta[b'coarse_label_names']
    labels = [label_names[index] for index in indices]
    print ("Selected categories: " + str(labels))


def getData(target_labels, show_info=False):
    """
    Gets data from folder 'data/' which should contain Cifar-100 database.
    Params:
        target_labels: list of course label indices included in the final dataset
    """

    data = unpickle("data/train")

    x_train = data[b'data']
    y_train = data[b'coarse_labels']

    # Select indices of labels that are in the target labels
    selected_indices_training = [idx for idx, label in enumerate(y_train) if label in target_labels]

    # Select only images from target categories to input
    x_input = x_train[selected_indices_training]
    
    if show_info:
        # Output selected categories
        printLabelNames(target_labels)
    
        # Images in selected data
        print ("Images in selected categories: " + str(len(x_input)))
    
    return x_input

data = getData([0], show_info=True)