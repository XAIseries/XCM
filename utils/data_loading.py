import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] ---> [0,1,2]


    Parameters
    ----------
    y_train: array
        Labels of the train set

    y_test: array
        Labels of the test set


    Returns
    -------
    new_y_train: array
        Transformed y_train array

    new_y_test: array
        Transformed y_test array
    """

    # Initiate the encoder
    encoder = LabelEncoder()

    # Concatenate train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)

    # Fit the encoder
    encoder.fit(y_train_test.ravel())

    # Transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test.ravel())

    # Resplit the train and test
    new_y_train = new_y_train_test[0 : len(y_train)]
    new_y_test = new_y_train_test[len(y_train) :]

    return new_y_train, new_y_test


def import_data(dataset, log=print):
    """
    Load and preprocess train and test sets


    Parameters
    ----------
    dataset: string
        Name of the dataset


    Returns
    -------
    X_train: array
        Train set without labels

    y_train: array
        Labels of the train set encoded

    X_test: array
        Test set without labels

    y_test: array
        Labels of the test set encoded

    y_train_nonencoded: array
        Labels of the train set non-encoded

    y_test_nonencoded: array
        Labels of the test set non-encoded
    """

    # Load train and test sets
    X_train = np.load("./datasets/" + dataset + "/X_train.npy")
    y_train = np.load("./datasets/" + dataset + "/y_train.npy")
    X_test = np.load("./datasets/" + dataset + "/X_test.npy")
    y_test = np.load("./datasets/" + dataset + "/y_test.npy")

    # Transform to continuous labels
    y_train, y_test = transform_labels(y_train, y_test)
    y_train_nonencoded, y_test_nonencoded = y_train, y_test

    # One hot encoding of the labels
    enc = OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # Reshape data to match 2D convolution filters input shape
    X_train = np.reshape(
        np.array(X_train),
        (X_train.shape[0], X_train.shape[2], X_train.shape[1], 1),
        order="C",
    )
    X_test = np.reshape(
        np.array(X_test),
        (X_test.shape[0], X_test.shape[2], X_test.shape[1], 1),
        order="C",
    )

    log("\nDataset" + " " + dataset + " " + "loaded")
    log("Training set size: {0}".format(len(X_train)))
    log("Testing set size: {0}".format(len(X_test)))

    return X_train, y_train, X_test, y_test, y_train_nonencoded, y_test_nonencoded
