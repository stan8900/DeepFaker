# GAN Image Generator

This repository contains an implementation of a Generative Adversarial Network (GAN) using PyTorch. The network is designed to generate realistic images by training a generator and a discriminator in an adversarial setting.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Training](#training)
- [Results](#results)
- [References](#references)

## Project Overview

This project implements a basic Generative Adversarial Network (GAN) in PyTorch. The GAN consists of two neural networks:
- **Generator**: Takes random noise as input and generates images.
- **Discriminator**: Distinguishes between real and generated images.

The objective is for the generator to produce images that are indistinguishable from real images, while the discriminator tries to correctly classify real vs. fake images. Over time, both networks improve, resulting in the generation of realistic images.

## Architecture

### Generator
The generator takes a random noise vector as input and transforms it into a 64x64 RGB image using a series of transpose convolutional layers, batch normalization, and ReLU activations. The final layer outputs an image using the `tanh` activation function.

- Input: 100-dimensional random noise vector
- Output: 64x64 RGB image

### Discriminator
The discriminator is a convolutional neural network that classifies whether an image is real or generated. It uses Leaky ReLU activations and a sigmoid function to output a probability (real or fake).

- Input: 64x64 RGB image
- Output: Probability (real or fake)

## Installation

### Requirements

- Python 3.x
- PyTorch
- Torchvision
- CUDA (optional, for GPU support)

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/your-username/gan-image-generator.git
   cd gan-image-generator
   ```
2. Install the dependencies:
```
pip install torch torchvision
```

## Training
To train the GAN, use the following command:

```
python train.py
```

Training Parameters
- batch_size: 128 (default)
- image_size: 64x64
- nz: Size of the input noise vector (default: 100)
- num_epochs: 25 (default)
- learning_rate: 0.0002 (default for both generator and discriminator)
During training, the discriminator and generator are updated in each iteration, and loss values are printed for both networks.
## Saving Results
Generated images are saved every 5 epochs under the following file structure:
```
/output
    fake_samples_epoch_{epoch_number}.png
```

## Results
The model generates 64x64 images after training for a few epochs. Here are some sample images:

Epoch 5:

Epoch 25:

## References
PyTorch Documentation: https://pytorch.org/get-started/locally/


DCGAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


