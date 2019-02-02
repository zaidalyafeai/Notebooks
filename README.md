# Notebooks

## Training pix2pix [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf_pix2pix.ipynb) 

This notebook shows a simple pipeline for training pix2pix on a simple dataset.

## One Place [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/ONePlace.ipynb)

This notebook shows how to train, test then deploy models in the browser directly from one notebook. We use a simple XOR example to prove this simple concept.


## TPU vs GPU [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/GPUvsTPU.ipynb)
Google recently allowed training on TPUs for free on colab. This notebook explains how to enable TPU training. Also, it reports some benchmarks using mnist dataset by comparing TPU and GPU performance.

## Keras Custom Data Generator [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Custom_Data_Generator_in_Keras.ipynb)
This notebook shows to create a custom data genertor in keras.

## Eager Execution and Gradient [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Eager_Execution_Gradient_.ipynb)

As we know that TenosrFlow works with static graphs. So, first you have to create the graph then execute it later. This makes debugging a bit complicated. With Eager Execution you can now evalute operations directly without creating a session. 

## Eager Execution Enabled [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Eager_Execution_Enabled.ipynb)

In this notebook I explain different concepts in eager execution. I go over variables, ops, gradients, custom gradients, callbacks, metrics and creating models with tf.keras and saving/restoring them. 

## Sketcher [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Sketcher.ipynb)

Create a simple app to recognize 100 drawings from the quickdraw dataset. A simple CNN model is created and served to deoploy in the browser to create a sketch recognizer app. 

## QuickDraw10 [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/QuickDraw10.ipynb)

In this notebook we provide QuickDraw10 as an alternative for MNIST. A script is provided to download and load a preprocessed dataset for 10 classes with training and testing split. Also, a simple CNN model is implemented for training and testing. 

## Autoencoders [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/AutoEncoders.ipynb)

Autoencoders consists of two structures: the encoder and the decoder. The encoder network downsamples the data into lower dimensions and the decoder network reconstructs the original data from the lower dimension representation. The lower dimension representation is usually called latent space representation. 

## Weight Transfer [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/WeightTransfer.ipynb)

In this tutorial we explain how to transfer weights from a static graph model built with TensorFlow to a dynamic graph built with Keras. We will first train a model using Tensorflow then we will create the same model in keras and transfer the trained weights between the two models. 

## BigGan [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/BigGan.ipynb)

Create some cool gifs by interpolation in the latent space of the BigGan model. The model is imported from tensorflow hub. 

## BigGanEx [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/BigGanEx.ipynb)
In this notebook I give a basic introduction to bigGans. I also, how to interpolate between z-vector values. Moreover, I show the 
results of multiple experiments I made in the latent space of BigGans. 

## Mask R-CNN [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Mask_RCNN.ipynb)
In this notebook a pretrained Mask R-CNN model is used to predict the bounding box and the segmentation mask of objects. I used this notebook to create the dataset for training the pix2pix model. 

## QuickDraw Strokes [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Strokes_QuickDraw.ipynb)
A notebook exploring the drawing data of quickdraw. I also illustrate how to make a cool animation of the drawing process in 
colab. 

## U-Net [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/unet.ipynb)
The U-Net model is a simple fully  convolutional neural network that is used for binary segmentation i.e foreground and background pixel-wise classification. In this notebook we use it to segment cats and dogs from arbitrary images. 

## Localizer [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Localizer.ipynb)
A simple CNN with a regression branch to predict bounding box parameters. The model is trained on a dataset 
of dogs and cats with bounding box annotations around the head of the pets. 

## Classification and Localization [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf_ClassficationLocalization.ipynb)
We create a simple CNN with two branches for classification and locazliation of cats and dogs. 

## Transfer Learning [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf_TransferLearning.ipynb.ipynb)
A notebook about using Mobilenet for transfer learning in TensorFlow. The model is very fast and achieves 97% validation accuracy on a binary classification dataset. 

