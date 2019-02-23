# Notebooks

<table class="tg">
  <tr>
    <th class="tg-yw4l"><b>Name</b></th>
    <th class="tg-yw4l"><b>Description</b></th>
    <th class="tg-yw4l"><b>Category</b></th>
    <th class="tg-yw4l"><b> Link </b></th>
  </tr>
  <tr>
    <td class="tg-yw4l">Training pix2pix</td>
    <td class="tg-yw4l">This notebook shows a simple pipeline for training pix2pix on a simple dataset. Most of the code is based on <a href = 'https://github.com/affinelayer/pix2pix-tensorflow' >this implementation</a>. </td>
    <td class="tg-yw4l">GAN</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf_pix2pix.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '800px' >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">One Place</td>
    <td class="tg-yw4l">This notebook shows how to train, test then deploy models in the browser directly from one notebook. We use a simple XOR example to prove this simple concept.</td>
    <td class="tg-yw4l">Deployment</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/ONePlace.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
   <tr>
    <td class="tg-yw4l">TPU vs GPU</td>
    <td class="tg-yw4l">Google recently allowed training on TPUs for free on colab. This notebook explains how to enable TPU training. Also, it reports some benchmarks using mnist dataset by comparing TPU and GPU performance.</td>
    <td class="tg-yw4l">TPU</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/GPUvsTPU.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
   <tr>
    <td class="tg-yw4l">Keras Custom Data Generator</td>
    <td class="tg-yw4l">This notebook shows to create a custom data genertor in keras.</td>
    <td class="tg-yw4l">Data Generatation</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Custom_Data_Generator_in_Keras.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Eager Execution (1)</td>
    <td class="tg-yw4l">As we know that TenosrFlow works with static graphs. So, first you have to create the graph then execute it later. This makes debugging a bit complicated. With Eager Execution you can now evalute operations directly without creating a session. </td>
    <td class="tg-yw4l">Dynamic Graphs </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Eager_Execution_Gradient_.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Eager Execution (2)</td>
    <td class="tg-yw4l">In this notebook I explain different concepts in eager execution. I go over variables, ops, gradients, custom gradients, callbacks, metrics and creating models with tf.keras and saving/restoring them.</td>
    <td class="tg-yw4l">Dynamic Graphs </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Eager_Execution_Enabled.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Sketcher</td>
    <td class="tg-yw4l">Create a simple app to recognize 100 drawings from the quickdraw dataset. A simple CNN model is created and served to deoploy in the browser to create a sketch recognizer app. </td>
    <td class="tg-yw4l"> Deployment </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Sketcher.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">QuickDraw10</td>
    <td class="tg-yw4l">In this notebook we provide QuickDraw10 as an alternative for MNIST. A script is provided to download and load a preprocessed dataset for 10 classes with training and testing split. Also, a simple CNN model is implemented for training and testing.  </td>
    <td class="tg-yw4l"> Data Preperation </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/QuickDraw10.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Autoencoders</td>
    <td class="tg-yw4l">Autoencoders consists of two structures: the encoder and the decoder. The encoder network downsamples the data into lower dimensions and the decoder network reconstructs the original data from the lower dimension representation. The lower dimension representation is usually called latent space representation.  </td>
    <td class="tg-yw4l"> Auto-encoder </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/AutoEncoders.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
   <tr>
    <td class="tg-yw4l">Weight Transfer</td>
    <td class="tg-yw4l">In this tutorial we explain how to transfer weights from a static graph model built with TensorFlow to a dynamic graph built with Keras. We will first train a model using Tensorflow then we will create the same model in keras and transfer the trained weights between the two models. </td>
    <td class="tg-yw4l"> Weights Save and Load</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/WeightTransfer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">BigGan (1)</td>
    <td class="tg-yw4l">Create some cool gifs by interpolation in the latent space of the BigGan model. The model is imported from tensorflow hub. 
 </td>
    <td class="tg-yw4l"> GAN</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/BigGAN.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">BigGan (2)</td>
    <td class="tg-yw4l">In this notebook I give a basic introduction to bigGans. I also, how to interpolate between z-vector values. Moreover, I show the  results of multiple experiments I made in the latent space of BigGans. 
 </td>
    <td class="tg-yw4l"> GAN </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/BigGanEx.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
   <tr>
    <td class="tg-yw4l">Mask R-CNN </td>
    <td class="tg-yw4l">In this notebook a pretrained Mask R-CNN model is used to predict the bounding box and the segmentation mask of objects. I used this notebook to create the dataset for training the pix2pix model. 
 </td>
    <td class="tg-yw4l"> Segmentation </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Mask_RCNN.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">QuickDraw Strokes</td>
    <td class="tg-yw4l"> A notebook exploring the drawing data of quickdraw. I also illustrate how to make a cool animation of the drawing process in colab. 
 </td>
    <td class="tg-yw4l"> Data Preperation </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Strokes_QuickDraw.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">U-Net</td>
    <td class="tg-yw4l"> The U-Net model is a simple fully  convolutional neural network that is used for binary segmentation i.e foreground and background pixel-wise classification. In this notebook we use it to segment cats and dogs from arbitrary images.  
 </td>
    <td class="tg-yw4l"> Segmentation</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/unet.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Localizer</td>
    <td class="tg-yw4l"> A simple CNN with a regression branch to predict bounding box parameters. The model is trained on a dataset 
of dogs and cats with bounding box annotations around the head of the pets.
 </td>
    <td class="tg-yw4l"> Object Localization</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/Localizer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Classification and Localization</td>
    <td class="tg-yw4l"> We create a simple CNN with two branches for classification and locazliation of cats and dogs. 
 </td>
    <td class="tg-yw4l"> Classification, Localization</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf_ClassficationLocalization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Transfer Learning </td>
    <td class="tg-yw4l"> A notebook about using Mobilenet for transfer learning in TensorFlow. The model is very fast and achieves 97% validation accuracy on a binary classification dataset.  
 </td>
    <td class="tg-yw4l"> Transfer Learning</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf_TransferLearning.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Hand Detection </td>
    <td class="tg-yw4l"> 
In this task we want to localize the right and left hands for each person that exists in a single frame. It acheives around 0.85 IoU.   
 </td>
    <td class="tg-yw4l"> Detection</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf_handBbox_esitmation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Face Detection </td>
    <td class="tg-yw4l"> 
In this task we used a simple version of SSD for face detection. The model was trained on less than 3K images using TensorFlow with eager execution 
 </td>
    <td class="tg-yw4l"> Detection</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/tf-Face-SSD.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" >
</a></td>
  </tr>
</table>


