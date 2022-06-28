# XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification
This repository contains the Keras implementations of XCM, XCM-Seq and MTEX-CNN networks 
as described in the paper [XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification](https://hal.inria.fr/hal-03469487/document).

![Alt-Text](/images/XCM.png)

## Requirements
Networks have been implemented in Python 3.8 with the following packages:
* keras
* matplotlib
* numpy
* pandas
* pyyaml
* scikit-learn
* seaborn
* tensorflow

## Usage
Run `main.py` with the following argument:

* configuration: name of the configuration file (string)

```
python main.py --config configuration/config.yml
```

The current configuration file provides an example of classification with XCM on the Basic Motions UEA dataset 
with the configuration presented in the paper, and an example of a heatmap from Grad-CAM for the
first MTS of the test set. 

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