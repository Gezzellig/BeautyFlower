import matplotlib.pyplot as plt
import numpy as np

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

data = unpickle("data/train")



x_train = data[b'data']
y_train = data[b'coarse_labels']

# All label categories we want from the cifar dataset (0 = aquatic mammals)
target_labels = [0]

# Output selected categories
printLabelNames(target_labels)

# Select indices of labels that are in the target labels
selected_indices_training = [idx for idx, label in enumerate(y_train) if label in target_labels]



x_input = x_train[selected_indices_training]

# Images in selected data
print ("Images in selected categories: " + str(len(x_input)))