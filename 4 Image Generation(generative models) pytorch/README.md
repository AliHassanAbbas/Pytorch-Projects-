# Image Generation with a Generative Adversarial Network (GAN) in PyTorch

This notebook implements a **fully connected, vanilla GAN** that learns to generate handwritten digit-like images using the MNIST dataset. It defines the generator and discriminator as separate PyTorch modules and trains them adversarially.

## Notebook

[Open 0_Image_Generation_with_Gans_in_pytorch.ipynb](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/4%20Image%20Generation%28generative%20models%29%20pytorch/0_Image_Generation_with_Gans_in_pytorch.ipynb)

## Model overview

A GAN consists of two networks trained in opposition:

- **Generator (G):** maps a random latent vector to a synthetic image.
- **Discriminator (D):** estimates whether an input image is real (from MNIST) or generated.

The generator uses fully connected layers:

`100 → 256 → 512 → 1024 → 784`

with LeakyReLU hidden activations and Tanh output. The discriminator uses:

`784 → 1024 → 512 → 256 → 1`

with LeakyReLU, Dropout and Sigmoid output.

This is a dense, fully connected GAN operating on flattened 28×28 grayscale images; it is not a convolutional DCGAN.

## Dataset and preprocessing

- Dataset: MNIST training split (downloaded through `torchvision.datasets.MNIST`)
- Image size: 28×28 grayscale (784 flattened features)
- Transform: `ToTensor()` followed by normalization with mean 0.5 and standard deviation 0.5
- Batch size: 64
- Latent dimension: 100
- Device: CUDA when available, otherwise CPU
- Random seeds are set for NumPy, PyTorch and Python's `random` module.

## Training configuration

| Setting | Value |
|---|---:|
| Epochs | 100 |
| Batch size | 64 |
| Latent dimension | 100 |
| Loss | Binary Cross Entropy (`nn.BCELoss`) |
| Optimizers | Separate Adam optimizers for G and D |
| Learning rate | 0.0002 |
| Adam betas | (0.5, 0.999) |
| Sample interval | Every 5 epochs |

The discriminator is trained to classify real images as 1 and generated images as 0. The generator is trained to make generated images receive the real label from the discriminator. Generated image grids are saved under `images/gan/` during training.

## Workflow covered

1. Load and normalize MNIST images with TorchVision
2. Create a shuffled PyTorch `DataLoader`
3. Sample random latent vectors
4. Implement generator and discriminator using `nn.Module` and `nn.Sequential`
5. Define adversarial binary-cross-entropy objectives
6. Configure independent Adam optimizers
7. Update discriminator and generator in alternating steps
8. Use `torch.no_grad()` / detach to control gradient flow during discriminator training and sample generation
9. Track losses and elapsed epoch time
10. Save and visualize generated image grids

## Run

Open the notebook in Google Colab or Jupyter and execute cells in order. The notebook uses PyTorch, TorchVision, NumPy and Matplotlib. MNIST is downloaded automatically by TorchVision. A CUDA GPU is optional; the code selects CUDA when available.

## Technical skills demonstrated

- Generative Adversarial Networks and adversarial optimization
- Designing separate generator and discriminator networks
- PyTorch custom modules, sequential layers and activation functions
- Latent-space sampling and noise-to-image generation
- Tensor reshaping and flattened image representations
- Dataset loading, preprocessing and normalization with TorchVision
- Binary cross-entropy objectives and Adam optimization
- Alternating optimization of two neural networks
- Gradient management with `zero_grad()`, `backward()`, `step()`, `detach()` and `torch.no_grad()`
- Reproducibility through random seed configuration
- Monitoring training losses and visualizing generated samples

## Implementation note

The notebook's inner training loop calls `next(iter(iterator))` repeatedly rather than iterating once over the existing DataLoader iterator. This repeatedly creates fresh iterators and draws a new first batch each time, instead of traversing one complete shuffled pass through the dataset per epoch. For conventional epoch semantics, use `for x, _ in iterator:` and perform both discriminator and generator updates for each batch. The README documents the notebook as written and does not silently change its training code.

## Scope

This is an educational vanilla GAN experiment on MNIST. It demonstrates the adversarial training mechanism and image-sample generation; it should not be described as a state-of-the-art image generator or as a convolutional GAN.

## Author

Ali Hassan Abbas | [GitHub](https://github.com/AliHassanAbbas)
