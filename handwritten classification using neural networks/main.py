import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from project2 import *

#######################################################################
# 1. Load data 
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()

# Convert data to binary classification and shuffle them
train_x, train_y, test_x, test_y = dataset_binary_label(train_x, train_y, test_x, test_y, label=5.)
train_x, train_y = shuffle_data(train_x, train_y)
test_x, test_y = shuffle_data(test_x, test_y)

# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])
list=[3,2,5]
params=L_layer_model(np.transpose(train_x),train_y,list)

predict(np.transpose(test_x),test_y,params)
