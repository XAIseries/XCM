# XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification
This repository contains the Keras implementations of XCM, XCM-Seq and MTEX-CNN networks 
as described in the paper [XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification](https://hal.inria.fr/hal-03469487/document).

![Alt-Text](/images/XCM.png)

## Requirements
Networks have been implemented in Python 3.8 with the following packages:
* Keras
* Numpy
* Pandas
* Scikit-Learn
* TensorFlow

## Usage
Run `main.py` with the following arguments:

* dataset: name of the dataset (string)
* model_name: name of the classifier (string)
* batch_size: batch size used to train the network (integer)
* window_size: percentage of the time series length expected to be interesting to extract discriminative features - only for XCM and XCM-Seq (float)

```
python main.py dataset model_name batch_size window_size
```

## Example 
In this section, we provide an example of classification with XCM on the Basic Motions UEA dataset with the configuration presented in the paper.

```
python main.py BasicMotions XCM 32 0.2
```

## Citation
```
@article{Fauvel21XCM,
  author = {Fauvel, K. and T. Lin and V. Masson and E. Fromont and A. Termier},
  title = {XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification},
  journal = {Mathematics},
  year = {2021},
  volume = {9},
  number = {23}
}
```