import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


def shuffle_data(X, Y):

    assert (X.shape[0] == Y.shape[0])
    shuffle_index = np.arange(X.shape[0])
    np.random.shuffle(shuffle_index)
    np.random.shuffle(shuffle_index)
    np.random.shuffle(shuffle_index)
    np.random.shuffle(shuffle_index)
    new_X = X[shuffle_index]
    new_Y = Y[shuffle_index]
    return new_X, new_Y


def dataset_binary_label(train_x, train_y, test_x, test_y, label=5.):
    training_count = 300
    test_count = 30
    labels = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    labels.remove(label)
    negative_train_x, negative_test_x = None, None
    for error_label in labels:
        (filtered_train_x, filtered_train_y, filtered_test_x, filtered_test_y) = \
            extract_training_and_test_examples_with_labels(
                train_x, train_y, test_x, test_y, [error_label], training_count, test_count)
        if negative_test_x is None:
            negative_train_x = filtered_train_x
            negative_test_x = filtered_test_x
        else:
            negative_train_x = np.append(
                negative_train_x, filtered_train_x, axis=0)
            negative_test_x = np.append(
                negative_test_x, filtered_test_x, axis=0)

    positive_train_x, _, positive_test_x, _ = \
        extract_training_and_test_examples_with_labels(
            train_x, train_y, test_x, test_y, [label], training_count, test_count)

    nb_p_train = positive_train_x.shape[0]
    nb_p_test = positive_test_x.shape[0]
    nb_n_train = negative_train_x.shape[0]
    nb_n_test = negative_test_x.shape[0]

    label_p_train = np.ones((nb_p_train,))
    label_p_test = np.ones((nb_p_test,))
    label_n_train = np.ones((nb_n_train,))*(-1)
    label_n_test = np.ones((nb_n_test,))*(-1)

    train_x = np.append(positive_train_x, negative_train_x, axis=0)
    test_x = np.append(positive_test_x, negative_test_x, axis=0)

    train_y = np.append(label_p_train, label_n_train)
    test_y = np.append(label_p_test, label_n_test)

    return train_x, train_y, test_x, test_y


def plot_images(X):
    if X.ndim == 1:
        X = np.array([X])
    num_images = X.shape[0]
    num_rows = math.floor(math.sqrt(num_images))
    num_cols = math.ceil(num_images/num_rows)
    for i in range(num_images):
        reshaped_image = X[i, :].reshape(28, 28)
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(reshaped_image, cmap=cm.Greys_r)
        plt.axis('off')
    plt.show()


def pick_examples_of(X, Y, labels, total_count):
    bool_arr = None
    for label in labels:
        bool_arr_for_label = (Y == label)
        if bool_arr is None:
            bool_arr = bool_arr_for_label
        else:
            bool_arr |= bool_arr_for_label
    filtered_x = X[bool_arr]
    filtered_y = Y[bool_arr]
    return (filtered_x[:total_count], filtered_y[:total_count])


def extract_training_and_test_examples_with_labels(train_x, train_y, test_x, test_y, labels, training_count, test_count):
    filtered_train_x, filtered_train_y = pick_examples_of(
        train_x, train_y, labels, training_count)
    filtered_test_x, filtered_test_y = pick_examples_of(
        test_x, test_y, labels, test_count)
    return (filtered_train_x, filtered_train_y, filtered_test_x, filtered_test_y)


def write_pickle_data(data, file_name):
    f = gzip.open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()


def read_pickle_data(file_name):
    f = gzip.open(file_name, 'rb')
    data = pickle.load(f, encoding='latin1')
    f.close()
    return data


def get_MNIST_data():
    """
    Reads mnist dataset from file

    Returns:
        train_x - 2D Numpy array (n, d) where each row is an image
        train_y - 1D Numpy array (n, ) where each row is a label
        test_x  - 2D Numpy array (n, d) where each row is an image
        test_y  - 1D Numpy array (n, ) where each row is a label

    """
    train_set, valid_set, test_set = read_pickle_data(
        './Datasets/mnist.pkl.gz')
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    train_x = np.vstack((train_x, valid_x))
    train_y = np.append(train_y, valid_y)
    test_x, test_y = test_set
    return (train_x, train_y, test_x, test_y)


def load_train_and_test_pickle(file_name):
    train_x, train_y, test_x, test_y = read_pickle_data(file_name)
    return train_x, train_y, test_x, test_y

# returns the feature set in a numpy ndarray


def load_CSV(filename):
    stuff = np.asarray(np.loadtxt(open(filename, 'rb'), delimiter=','))
    return stuff
