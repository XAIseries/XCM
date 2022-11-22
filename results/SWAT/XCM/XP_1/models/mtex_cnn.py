from keras.models import Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Reshape
from keras.layers.convolutional import Conv1D, Conv2D
from keras import regularizers


def mtex_cnn(input_shape, n_class):
    """
    MTEX-CNN model


    Parameters
    ----------
    input_shape: array
        Input shape array

    n_class: integer
        Number of classes


    Returns
    -------
    model: model
        MTEX-CNN Model
    """

    # Input layer shape
    n = input_shape[0]
    k = input_shape[1]
    input_layer = Input(shape=(n, k, 1))

    # 2D convolution layers
    a = Conv2D(
        filters=64,
        kernel_size=(8, 1),
        strides=(2, 1),
        padding="same",
        input_shape=(n, k, 1),
        name="2D_1",
    )(input_layer)
    a = Activation("relu", name="2D_1_Activation")(a)
    a = Dropout(0.4)(a)
    a = Conv2D(
        filters=128, kernel_size=(6, 1), strides=(2, 1), padding="same", name="2D_2"
    )(a)
    a = Activation("relu", name="2D_2_Activation")(a)
    a = Dropout(0.4)(a)
    a = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), name="2D_Reduced")(a)
    a = Activation("relu", name="2D_Reduced_Activation")(a)
    x = Reshape((int(n / 4), k))(a)

    # 1D convolution layer in sequence
    b = Conv1D(
        filters=128, kernel_size=4, strides=2, input_shape=(int(n / 4), k), name="1D"
    )(x)
    b = Activation("relu", name="1D_Activation")(b)
    y = Dropout(0.4)(b)

    # FCN for classification
    z = Flatten()(y)
    z = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.2))(z)
    output_layer = Dense(n_class, activation="softmax")(z)

    model = Model(input_layer, output_layer)

    print("MTEX-CNN Model Loaded")
    return model
