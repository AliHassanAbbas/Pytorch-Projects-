# Image Generation with GANs in PyTorch

This folder contains five educational PyTorch notebooks exploring several Generative Adversarial Network (GAN) architectures and objectives. The projects progress from a fully connected GAN on MNIST to convolutional GAN variants trained on CIFAR-10.

All notebooks define their own generator and discriminator/critic, training loops, optimizers and image-sampling routines. They are standalone experiments rather than one shared software package.

## Projects

### 1. Vanilla GAN — MNIST

[Open notebook](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/0_Image_Generation_with_Gans_in_pytorch.ipynb)

A fully connected GAN that learns to generate 28×28 grayscale handwritten-digit images.

- **Dataset:** MNIST training split
- **Generator:** fully connected layers 100 → 256 → 512 → 1024 → 784, LeakyReLU hidden activations and Tanh output
- **Discriminator:** fully connected network with LeakyReLU, Dropout(0.3) and Sigmoid output
- **Objective:** Binary Cross Entropy (BCE)
- **Optimizers:** separate Adam optimizers, learning rate 0.0002 and betas (0.5, 0.999)
- **Configuration:** batch size 64, latent dimension 100, 100 epochs
- **Implementation details:** flattened image tensors, Gaussian latent vectors, alternating generator/discriminator updates and generated sample grids

### 2. Deep Convolutional GAN (DCGAN) — CIFAR-10

[Open notebook](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/1_Image_Generation_with_DCGAN_in_pytorch.ipynb)

A convolutional GAN that generates 64×64 RGB images from 100-dimensional Gaussian noise.

- **Dataset:** CIFAR-10 training split, resized to 64×64
- **Generator:** ConvTranspose2d layers progressively upsample a [batch, 100, 1, 1] latent tensor; BatchNorm and ReLU in intermediate stages; Tanh output
- **Discriminator:** Conv2d layers downsample RGB images; BatchNorm and LeakyReLU; Sigmoid probability output
- **Objective:** BCE
- **Optimizers:** separate Adam optimizers, learning rate 0.0002 and betas (0.5, 0.999)
- **Configuration:** batch size 64, latent dimension 100, 20 epochs
- **Implementation details:** DCGAN-style convolutional weight initialization and generated-image grids

### 3. Least Squares GAN (LSGAN) — CIFAR-10

[Open notebook](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/2_Image_Generation_with_Least_Squares_GAN_in_pytorch.ipynb)

An LSGAN experiment that reuses a DCGAN-style convolutional generator and discriminator but changes the adversarial objective from BCE to least-squares losses.

- **Dataset and image size:** CIFAR-10 resized to 64×64 RGB
- **Architecture:** convolutional generator/discriminator with transposed convolutions, BatchNorm, ReLU and LeakyReLU
- **Objective:** least-squares losses computed with mean squared differences from real/fake target values; discriminator output is not passed through Sigmoid
- **Optimizers:** separate Adam optimizers, learning rate 0.0002 and betas (0.5, 0.999)
- **Configuration:** batch size 64, latent dimension 100, 15 epochs
- **Key comparison:** changing the loss formulation while retaining a DCGAN-like network design

### 4. Wasserstein GAN (WGAN) — CIFAR-10

[Open notebook](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/3_Image_Generation_with_WGAN_in_pytorch.ipynb)

A weight-clipped WGAN implementation that uses the Wasserstein critic objective rather than a sigmoid-based binary classifier.

- **Dataset and image size:** CIFAR-10 resized to 64×64 RGB
- **Architecture:** convolutional generator and critic; critic outputs an unrestricted score (no Sigmoid)
- **Critic objective:** mean score on generated images minus mean score on real images
- **Generator objective:** negative mean critic score on generated images
- **Optimizers:** RMSprop for both networks, learning rate 0.00005
- **WGAN training strategy:** five critic updates per generator update (D_UPDATES = 5)
- **Weight constraint:** critic parameters are clamped to [-0.01, 0.01] after critic updates
- **Configuration:** batch size 64, latent dimension 100, 15 epochs
- **Implementation details:** custom convolutional weight initialization and generated sample grids

This notebook implements the original weight-clipping WGAN training approach, not WGAN-GP; it does not implement a gradient penalty.

### 5. Relativistic Average Least Squares GAN (RaLSGAN) — CIFAR-10

[Open notebook](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/4_Image_Generation_with_RLSGAN_in_pytorch.ipynb)

A relativistic-average least-squares adversarial objective using a DCGAN-style convolutional generator and discriminator. The discriminator compares real and generated scores relative to the opposing batch's mean score.

- **Dataset and image size:** CIFAR-10 resized to 64×64 RGB
- **Architecture:** convolutional generator and discriminator with BatchNorm, ReLU and LeakyReLU
- **Generator output:** raw image values; the Tanh layer is commented out in this notebook
- **Discriminator output:** unrestricted scores; the Sigmoid layer is commented out
- **Objective:** relativistic-average least-squares equations implemented directly using batch means and squared differences
- **Optimizers:** separate Adam optimizers, learning rate 0.0001 and betas (0.5, 0.999)
- **Configuration:** batch size 64, latent dimension 100, 100 epochs
- **Implementation details:** custom convolutional weight initialization and per-epoch sample grids

**Code note:** BCEWithLogitsLoss is instantiated in this notebook but is not used by the actual RaLSGAN loss calculations. The explicit relativistic-average least-squares expressions drive training.

## Shared implementation patterns

Across these notebooks, the code demonstrates:

- Building custom models with torch.nn.Module and nn.Sequential
- Designing fully connected and convolutional generators/discriminators
- Mapping Gaussian latent noise into synthetic image tensors
- Using ConvTranspose2d for upsampling and Conv2d for learned downsampling
- Batch normalization, LeakyReLU, ReLU, Tanh and Sigmoid activations
- Implementing BCE, least-squares, Wasserstein and relativistic-average least-squares objectives
- Alternating optimization of generator and discriminator/critic networks
- Managing gradient flow with torch.no_grad(), .detach(), and separate optimizer steps
- Custom weight initialization, optimizer configuration and parameter constraints
- Reproducibility through Python, NumPy and PyTorch random seeds
- Selecting CUDA when available, timing epochs, monitoring losses and visualizing generated samples

## Important implementation notes

These are educational notebooks, so review the code carefully before treating them as production training pipelines.

- The training loops commonly call next(iter(iterator)) inside the loop. This constructs a new DataLoader iterator each time rather than consuming one persistent iterator through the dataset. Consequently, the code does not perform a conventional full pass over distinct batches in each epoch. A production-quality revision should create one iterator per epoch and iterate over it directly.
- Some notebooks create real/fake label tensors using the configured batch size. If the final batch is smaller, labels should be created using the actual batch dimension (or the loader should use drop_last=True).
- The WGAN notebook uses weight clipping, the original WGAN constraint, and should not be described as WGAN-GP.
- The RaLSGAN notebook defines a BCE-with-logits loss object but trains with its explicit relativistic least-squares equations instead.

## Running the notebooks

1. Open a notebook in Jupyter or Google Colab.
2. Install PyTorch, TorchVision, NumPy and Matplotlib if they are not already available.
3. Run the cells in order. The notebooks download MNIST or CIFAR-10 through TorchVision.
4. A CUDA GPU is used automatically when available; otherwise, the code falls back to CPU.
5. Generated image grids are saved under the corresponding images/ subdirectory.

Exact runtime and memory requirements depend on the selected hardware and notebook.

## Technical expertise demonstrated

**PyTorch and deep learning:** custom neural-network modules, tensor shape management, convolutional architectures, autograd, gradient control, optimizer configuration and training loops.

**Generative modelling:** GAN fundamentals, DCGAN, LSGAN, weight-clipped WGAN and RaLSGAN; latent-space sampling; adversarial objectives; generator/critic dynamics.

**Computer vision:** MNIST/CIFAR-10 input pipelines, image resizing and normalization, grayscale and RGB tensor handling, image synthesis and visualization.

**Experimentation:** random seeding, initialization strategies, training diagnostics, epoch timing, sample-grid generation and comparison of alternative loss functions.

## Author

Ali Hassan Abbas · [GitHub](https://github.com/AliHassanAbbas) · [Repository](https://github.com/AliHassanAbbas/Pytorch-Projects-)
