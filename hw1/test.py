import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io

def partition_mnist(data, labels):

    np.random.seed(10)
    permuted_data = np.random.permutation(data)

    np.random.seed(10)
    permuted_labels = np.random.permutation(labels)

    valiadation_data = permuted_data[:10000]
    training_data = permuted_data[10000:]

    valiadation_labels = permuted_labels[:10000]
    training_labels = permuted_labels[10000:]

    return training_data, training_labels, valiadation_data, valiadation_labels

data = np.load("/Users/Dom/Desktop/CS189/hw1/data/mnist-data.npz")
training_data = data["training_data"]
training_labels = data["training_labels"]
print(partition_mnist(training_data, training_labels))
print(training_data)
print(training_labels)