# PyTorch Projects and Deep Learning Notebooks

A broad collection of practical PyTorch learning projects and notebooks across computer vision, natural language processing, generative modelling and other deep-learning tasks.

## Areas represented
- Linear regression and image classification
- Sentiment analysis and sequence modelling
- Translation and language tasks
- Image generation and generative models
- Segmentation, detection and recognition topics
- Recommendation systems and time-series learning topics

## Repository structure
Projects are organized into topic-specific directories. Browse the folders and open the notebooks relevant to the task you want to study. Not every listed topic should be assumed to have a complete implementation; inspect each directory for available files.

## Run
Open an individual notebook in Jupyter or Google Colab. Each project may require different dependencies, datasets and hardware. Follow its setup and data-preparation cells.

## Scope
An educational collection, not one unified Python package. Model architectures, datasets, training procedures and evaluation methods vary by notebook.

## Author
Ali Hassan Abbas | [GitHub](https://github.com/AliHassanAbbas)

## Linear Regression with PyTorch

A hands-on linear regression notebook that predicts exam scores from hours studied. It covers NumPy data preparation, tensor conversion, model construction with `nn.Module` and `nn.Linear`, MSE loss, SGD, a manual training loop, evaluation/inference, visualization, prediction on a new input, and saving model parameters with `state_dict`.


[Read the project README](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/0%20Linear%20Regression/README.md) · [Open the notebook](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/0%20Linear%20Regression/0_Basic_Linear_Regression.ipynb)


## PyTorch Fundamentals Demonstrated

- Supervised learning fundamentals and linear regression
- NumPy data preparation and PyTorch tensors
- `nn.Module`, `nn.Linear`, parameters, weights and bias
- Forward passes, loss calculation, backpropagation and SGD updates
- Manual training loops and optimizer workflows
- Model evaluation, inference and gradient disabling
- Data visualization and prediction plots
- Saving and restoring model weights with `state_dict`


## Image Generation with a Vanilla GAN

A PyTorch implementation of a fully connected Generative Adversarial Network trained on MNIST. Includes separate generator and discriminator modules, latent-noise sampling, adversarial Binary Cross Entropy losses, alternating Adam optimization, gradient-flow management, and generated image visualization.

[Read the GAN project README](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/README.md) · [Open the GAN notebook](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/0_Image_Generation_with_Gans_in_pytorch.ipynb)

### Generative Modelling Skills Demonstrated

- GAN architecture and adversarial learning
- Generator/discriminator design with PyTorch
- Latent-space sampling and image generation
- MNIST preprocessing and tensor reshaping
- Binary Cross Entropy and Adam optimization
- Alternating optimization and gradient-flow control
- Reproducibility, loss monitoring and sample visualization


## Deep Convolutional GAN (DCGAN) — CIFAR-10

A from-scratch PyTorch implementation of a Deep Convolutional Generative Adversarial Network. The generator maps 100-dimensional Gaussian noise to 64×64 RGB images using transposed convolutions, while the discriminator uses convolutional layers to distinguish real CIFAR-10 images from generated samples. Includes Batch Normalization, custom weight initialization, ReLU/LeakyReLU/Tanh activations, Binary Cross Entropy objectives, separate Adam optimizers, gradient-flow management, and generated image grids.

[Read the image-generation README](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/README.md) · [Open the DCGAN notebook](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/1_Image_Generation_with_DCGAN_in_pytorch.ipynb)

### Additional Generative Modelling Skills

- Deep Convolutional GANs and convolutional image synthesis
- Transposed convolution and progressive spatial upsampling
- Convolutional feature extraction and downsampling
- Batch Normalization and DCGAN-style weight initialization
- Gaussian latent-noise sampling and RGB image generation
- Adversarial training with separate generator/discriminator optimizers
