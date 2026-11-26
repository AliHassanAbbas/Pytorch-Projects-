# Image Generation with GANs in PyTorch

This directory contains hands-on implementations of two foundational **Generative Adversarial Network (GAN)** approaches: a fully connected GAN and a Deep Convolutional GAN (DCGAN). Both models are implemented with PyTorch and trained from scratch.

## Projects

### 1. Vanilla GAN — MNIST

[Open 0_Image_Generation_with_Gans_in_pytorch.ipynb](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/0_Image_Generation_with_Gans_in_pytorch.ipynb)

A fully connected GAN that learns to generate 28×28 grayscale digit images from the MNIST dataset.

**Model design**
- Generator: fully connected layers mapping a 100-dimensional latent vector through hidden layers to 784 output values.
- Discriminator: fully connected classifier that distinguishes real MNIST images from generated samples.
- LeakyReLU hidden activations, Dropout in the discriminator, and Tanh generator output.

**Training setup**
- Dataset: MNIST training split
- Image representation: flattened 28×28 grayscale images
- Batch size: 64
- Latent dimension: 100
- Epochs: 100
- Loss: Binary Cross Entropy
- Optimizers: separate Adam optimizers for generator and discriminator
- Learning rate: 0.0002; betas: (0.5, 0.999)

### 2. Deep Convolutional GAN (DCGAN) — CIFAR-10

[Open 1_Image_Generation_with_DCGAN_in_pytorch.ipynb](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/1_Image_Generation_with_DCGAN_in_pytorch.ipynb)

A convolutional GAN that learns to generate 64×64 RGB images using the CIFAR-10 training dataset resized to 64×64.

**Architecture**
- **Generator:** accepts random latent noise shaped as [batch, 100, 1, 1] and uses a stack of transposed convolutions to progressively upsample it into 64×64 RGB images.
- **Discriminator:** uses convolutional layers to progressively downsample RGB images and output a real/fake probability.
- Batch Normalization is used in intermediate convolutional blocks.
- ReLU activations are used in the generator; LeakyReLU activations are used in the discriminator.
- Tanh is used at the generator output, matching the normalized image range.

**Training setup**
| Setting | Value |
|---|---:|
| Dataset | CIFAR-10 training split |
| Image size | 64 × 64 × 3 |
| Batch size | 64 |
| Latent dimension | 100 |
| Epochs | 20 |
| Loss | Binary Cross Entropy |
| Optimizers | Separate Adam optimizers |
| Learning rate | 0.0002 |
| Adam betas | (0.5, 0.999) |
| Sample interval | Every 5 epochs |
| Weight initialization | Normal initialization for convolution and BatchNorm layers |

Generated image grids are saved under `images/dcgan/` at the configured sample intervals.

## Concepts and workflow covered

1. Load image datasets using `torchvision.datasets`.
2. Resize, convert to tensors, and normalize image data.
3. Build shuffled batches with PyTorch `DataLoader`.
4. Sample random latent vectors from a Gaussian distribution.
5. Implement generator and discriminator networks with `nn.Module` and `nn.Sequential`.
6. Use dense layers in the vanilla GAN and convolutional layers in DCGAN.
7. Define adversarial objectives with `nn.BCELoss`.
8. Train the discriminator on real and generated images.
9. Train the generator to produce samples classified as real.
10. Manage gradient flow with `zero_grad()`, `backward()`, `step()`, `detach()`, and `torch.no_grad()`.
11. Initialize model weights and configure independent Adam optimizers.
12. Monitor losses and elapsed training time.
13. Save and visualize generated image grids.

## Technical skills demonstrated

- Generative Adversarial Networks (GANs) and DCGAN architecture
- Convolutional neural network design for image generation
- Transposed convolutions and learned upsampling
- Convolutional feature extraction and spatial downsampling
- Latent-space sampling and noise-to-image generation
- PyTorch custom model classes, sequential modules, and tensor operations
- Batch Normalization, ReLU, LeakyReLU, and Tanh
- Adversarial Binary Cross Entropy objectives
- Alternating optimization of two neural networks
- Custom weight initialization
- Gradient-flow control and training-loop implementation
- TorchVision dataset loading and image preprocessing
- GPU-aware training, reproducibility, loss monitoring, and visualization

## Run the notebooks

Open either notebook in Google Colab or Jupyter and execute its cells in order. The projects use PyTorch, TorchVision, NumPy, and Matplotlib. The datasets are downloaded automatically through TorchVision when required.

A CUDA-capable GPU is recommended, particularly for DCGAN training, but the notebooks select CPU when CUDA is unavailable.

## Implementation notes

- Both notebooks are educational implementations and document the code as currently written.
- In both training loops, `next(iter(iterator))` is called inside the loop. This creates a fresh DataLoader iterator for each batch request instead of traversing one persistent iterator through the dataset. For conventional epoch semantics, use `for x, _ in iterator:` and perform the discriminator and generator updates for each batch.
- The DCGAN notebook uses fixed-size real/fake label tensors of length 64. Its current repeated fresh-iterator pattern produces full batches in this setup; if the loop is refactored to traverse the DataLoader normally, handle a smaller final batch by creating labels based on the actual batch size.
- The notebooks demonstrate foundational GAN training; they do not claim state-of-the-art image quality or production-ready training stability.

## Author

Ali Hassan Abbas | [GitHub](https://github.com/AliHassanAbbas)
