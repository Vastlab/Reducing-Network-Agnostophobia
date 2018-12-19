# Reducing Network Agnostophobia

This Repository provides the code to reproduce all the results for the MNIST experiments in the paper "Reducing Network Agnostophobia" to be presented at NIPS 2018 as a Spotlight presentation.

## Datasets Preprocessing
All datasets and their preprocessing is provided in [MNIST/data_prep.py](https://github.com/Vastlab/ObjectoSphere/blob/master/MNIST/data_prep.py).
The preprocessing is largely limited to normalization of pixel values except for MNIST where we negate the pixel values before normalization, in order to achieve a consistent black on white background images for all datasets.
The datasets used are:
  * MNIST
  * NIST letters
  * Not MNIST
  * CIFAR
  * Devanagri

You may want to update the dataset locations in [MNIST/data_prep.py](https://github.com/Vastlab/ObjectoSphere/blob/master/MNIST/data_prep.py).

## Training
[MNIST/Mnist_Training.py](https://github.com/Vastlab/ObjectoSphere/blob/master/MNIST/Mnist_Training.py) provides the script to train models with all the loss functions used in different experiments. 
The random models used to initialize the networks trained in the paper are also provided at [MNIST/LeNet++/Random_Models](https://github.com/Vastlab/ObjectoSphere/tree/master/MNIST/LeNet%2B%2B/Random_Models).
The types of networks supported by this script contain:

1. Original LeNet/LeNet++ with Softmax loss.
2. LeNet/LeNet++ with Background Class.
3. LeNet/LeNet++ with Entropic OpenSet Loss.
4. LeNet/LeNet++ with ObjectoSphere Loss.

In the current training process, we run the training for the given network for 70 epochs and save only the model with the least validation loss.
The parameters used for training are provided as the default parameters for the script.
For details please refer to the help provided in [MNIST/Mnist_Training.py](https://github.com/Vastlab/ObjectoSphere/blob/master/MNIST/Mnist_Training.py)

The training and testing process for the OpenMax technique is provided in [MNIST/openmax.py](https://github.com/Vastlab/ObjectoSphere/blob/master/MNIST/openmax.py)

## Reproducing Results
All the visualizations for the MNIST experiments provided in the paper can be reproduced following the Jupyter Notebook at [MNIST/Fig_creator.ipynb](https://github.com/Vastlab/ObjectoSphere/blob/master/MNIST/Fig_creator.ipynb).
It utilizes the tools available in [Tools](https://github.com/Vastlab/ObjectoSphere/tree/master/Tools).
The [Tools/model_tools.py](https://github.com/Vastlab/ObjectoSphere/tree/master/Tools/model_tools.py) contains functions to define the network architecture, extract feature vectors from a specific layer of the network and preprocess data for implementing in the loss function.
The [Tools/visualizing_tools.py](https://github.com/Vastlab/ObjectoSphere/tree/master/Tools/visualizing_tools.py) contains the plotting functions for the two-dimensional plots and the histograms.
And, the [Tools/evaluation_tools.py](https://github.com/Vastlab/ObjectoSphere/tree/master/Tools/evaluation_tools.py) contains the plotting functions for the DIR curves in the paper as well as a function to write the results into a preliminary file.
Please refer to these files for a detailed understanding.
