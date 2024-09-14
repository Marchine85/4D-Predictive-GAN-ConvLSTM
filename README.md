3D Spatio-Temporal (4D) Predictions with ConvLSTM and GAN models
=====================================
This is the code for reproducing experiments from the master's thesis '3D Spatio-Temporal Predictions with ConvLSTM and GAN models' written at FH Technikum Vienna.

## Overview
__Abstract__<br>
*Forecasting video frames is a research topic for several years, where the general idea is to
use a set of video frames of the past and to predict the next event(s) that will happen in the
scene. Although a lot of publications are available and advances in this field have been made,
barely anything can be found on the topic of next frame predictions of 3D data, like the state
or topology change over time of objects. Applications for such models obviously exist, like
time-dependent physical simulations, Magnet Resonance Imaging, CT-scans, to name a few.
Therefore, in this work the two most commonly used approaches, ConvLSTM and GAN-based
networks, are trained and tested upon their applicability on 3D spatio-temporal predictions of
topological changes. The artificially generated academic dataset that is used, represents developing
cavities within a confined 3D grid space over time. The models are evaluated by means of
single next frame prediction and long-term predictive performance utilizing the Jaccard Distance
as a quality measure. Additionally, three different dataset formats are evaluated, representing
smaller and larger frame to frame transitions. The results show that larger frame steps didn’t
produced an obstacle for the models, but have even encouraged learning the transition from
one frame to the next in both. Out of the two approaches and setups tested, the ConvLSTM
model showed the best performance overall. This work shows that the 2D approaches are
applicable to the 3D space and delivers insights on how many future frames can be predicted
with an acceptable confidence, based on the two modelling approaches and the dataset under
consideration.*

__Dataset__<br>
<p align="center">
  <img src="imgsrc/FutureGAN_Framework.png">
  <br><br>
  We initialize both networks to take a set of 4 × 4 px resolution frames and output frames of the same resolution.
  During training, layers are added progressively for each resolution step. The resolution of the input frames matches the resolution of the current state of the networks.
</p>

## Prerequisites

- Python, NumPy, TensorFlow 2.10, SciPy, Matplotlib
- NVIDIA GPU

## Models

Configuration for all models is specified in a list of constants at the top of
the file. Two models should work "out of the box":

- `python gan_toy.py`: Toy datasets (8 Gaussians, 25 Gaussians, Swiss Roll). 
- `python gan_mnist.py`: MNIST

For the other models, edit the file to specify the path to the dataset in
`DATA_DIR` before running. Each model's dataset is publicly available; the
download URL is in the file.

- `python gan_64x64.py`: 64x64 architectures (this code trains on ImageNet instead of LSUN bedrooms in the paper)
- `python gan_language.py`: Character-level language model
- `python gan_cifar.py`: CIFAR-10
