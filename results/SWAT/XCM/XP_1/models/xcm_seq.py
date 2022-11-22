from keras.models import Model
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    GlobalAveragePooling1D,
    Input,
    Reshape,
)
from keras.layers.convolutional import Conv1D, Conv2D


def xcm_seq(input_shape, n_class, window_size, filters_num=128):
    """
    XCM-Seq model


    Parameters
    ----------
    input_shape: array
        Input shape array

    n_class: integer
        Number of classes

    window_size: float
        Time windows size, i.e. size of the subsequence of the MTS
        expected to be interesting to extract discriminative features

    filters_num: integer
        Number of filters
        Default value: 128


    Returns
    -------
    model: model
        XCM-Seq Model
    """

    # Input layer shape
    n = input_shape[0]
    k = input_shape[1]
    input_layer = Input(shape=(n, k, 1))

    # 2D convolution layers
    a = Conv2D(
        filters=int(filters_num),
        kernel_size=(int(window_size * n), 1),
        strides=(1, 1),
        padding="same",
        input_shape=(n, k, 1),
        name="2D",
    )(input_layer)
    a = BatchNormalization()(a)
    a = Activation("relu", name="2D_Activation")(a)
    a = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), name="2D_Reduced")(a)
    x = Activation("relu", name="2D_Reduced_Activation")(a)

    # 1D convolution layers in sequence
    b = Reshape((n, k))(x)
    b = Conv1D(
        filters=int(filters_num),
        kernel_size=int(window_size * n),
        strides=1,
        padding="same",
        name="1D",
    )(b)
    b = BatchNormalization()(b)
    b = Activation("relu", name="1D_Activation")(b)
    b = Conv1D(filters=1, kernel_size=1, strides=1, name="1D_Reduced")(b)
    y = Activation("relu", name="1D_Reduced_Activation")(b)

    # 1D convolution layer
    z = Conv1D(
        filters=filters_num,
        kernel_size=int(window_size * n),
        strides=1,
        padding="same",
        name="1D_Final",
    )(y)
    z = BatchNormalization()(z)
    z = Activation("relu", name="1D_Final_Activation")(z)

    # 1D global average pooling and classification
    z = GlobalAveragePooling1D()(z)
    output_layer = Dense(n_class, activation="softmax")(z)

    model = Model(input_layer, output_layer)

    print("XCM-Seq Model Loaded")
    return model
