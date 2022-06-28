import keras
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def grad_cam(data, model, layer_name, conv_type):
    """
    Grad-CAM output


    Parameters
    ----------
    data: array
        MTS sample

    model: model
        Trained model

    layer_name: string
        Name of the convolution layer

    conv_type: string
        Type of the convolution layer


    Returns
    -------
    heatmap: array
        Heatmap
    """

    # Get class-specific gradient information with respect to feature map activations of the convolution layer
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_layer_output, preds = grad_model(data)
        pred_index = tf.argmax(preds[0])
        pred_class = preds[:, pred_index]

    grads = tape.gradient(pred_class, conv_layer_output)

    # Compute a weighted combination between the feature maps
    if conv_type == "1D":
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        conv_layer_output = conv_layer_output[0]
        heatmap = conv_layer_output * pooled_grads
        heatmap = tf.reduce_mean(heatmap, axis=(1))
        heatmap = np.array(heatmap).reshape(1, heatmap.shape[0], 1, 1)
        heatmap = keras.layers.UpSampling2D(size=(1, data.shape[2]))(heatmap)
        heatmap = np.squeeze(heatmap)
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_layer_output = conv_layer_output[0]
        heatmap = conv_layer_output * pooled_grads
        heatmap = tf.reduce_mean(heatmap, axis=(2))

    # Keep positive values
    heatmap = np.maximum(heatmap, 0)
    return heatmap


def get_heatmap(
    configuration, xp_dir, model, X_train, X_test, y_train_nonencoded, y_test_nonencoded
):
    """
    Get the heatmap supporting a prediction


    Parameters
    ----------
    configuration: array
        Elements from the configuration file

    xp_dir: string
        Directory used to save the results of the experiment

    model: model
        Trained model

    X_train: array
        Train set without labels

    X_test: array
        Test set without labels

    y_train_nonencoded: array
        Labels of the train set non-encoded

    y_test_nonencoded: array
        Labels of the test set non-encoded
    """

    # Retrieve configuration
    set_explanation = configuration["set"]
    mts_sample_id = configuration["mts_sample_id"]
    layer_name = configuration["layer_name"]

    # Get MTS sample
    set_dict = {
        "train": [X_train, y_train_nonencoded],
        "test": [X_test, y_test_nonencoded],
    }
    mts_label = set_dict[set_explanation][1][mts_sample_id]
    mts_sample = set_dict[set_explanation][0][mts_sample_id]
    mts_sample = np.expand_dims(mts_sample, 0)

    # Get Grad-CAM output
    heatmap = grad_cam(mts_sample, model, layer_name, layer_name[:2])

    # Save and display the heatmap
    plt.figure(figsize=(25, 10))
    heatmap = np.swapaxes(normalize(heatmap), 0, 1)
    xticklabels = range(1, mts_sample.shape[1] + 1)
    yticklabels = range(1, mts_sample.shape[2] + 1)
    sns.heatmap(
        heatmap, xticklabels=xticklabels, yticklabels=yticklabels, cmap="RdBu_r"
    )
    plt.title(
        "Set: "
        + set_explanation
        + ", MTS ID: "
        + str(mts_sample_id)
        + ", Label: "
        + str(mts_label)
        + ", Prediction: "
        + str(np.argmax(model.predict(mts_sample), axis=1)[0])
        + ", Layer: "
        + layer_name
    )
    plt.savefig(
        xp_dir
        + "grad-cam/"
        + set_explanation
        + "_MTS_"
        + str(mts_sample_id)
        + "_layer_"
        + layer_name
        + ".png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
