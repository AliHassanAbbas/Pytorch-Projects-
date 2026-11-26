# Image Classification with PyTorch

A progression of hands-on image-classification notebooks, moving from a simple linear classifier to foundational convolutional neural network (CNN) architectures. The examples use PyTorch and Torchvision to build models, prepare image datasets, train classifiers, and evaluate their predictions.

## Notebooks

| # | Notebook | Main topic |
|---|---|---|
| 0 | [Logistic Regression](./0_Image_classification_with_Logistic_Regression.ipynb) | A linear baseline for multi-class image classification |
| 1 | [Neural Networks in PyTorch](./1_Classification_with_neural_networks_in_Pytorch.ipynb) | A feed-forward neural network (MLP) |
| 2 | [Advanced MLP](./2_Image_classification_with_MLP_%28Advanced%29_in_pytorch.ipynb) | Multi-layer perceptron image classification |
| 3 | [LeNet](./3_Image_classification_with_LeNet_in_pytorch.ipynb) | A classic convolutional neural network |
| 4 | [AlexNet](./4_Image_classification_with_AlexNet_in_pytorch.ipynb) | A landmark deep CNN architecture |
| 5 | [VGGNet](./5_Image_classification_with_VGGNET_in_pytorch.ipynb) | Deep networks built from repeated convolutional blocks |
| 6 | [ResNet](./6_Image_classification_with_ResNet_with_Pytorch.ipynb) | Residual learning and skip connections |

## Dataset and workflow

The first two notebooks use the **Fashion-MNIST** dataset through `torchvision.datasets`. Images are converted to tensors with `transforms.ToTensor()`, then provided in mini-batches using `torch.utils.data.DataLoader`.

The demonstrated training workflow includes:

1. Loading training and test splits.
2. Preparing image tensors and batches.
3. Defining models by subclassing `torch.nn.Module).
4. Running forward passes and calculating multi-class loss with `nn.CrossEntropyLoss`.
5. Updating model parameters with PyTorch optimizers (SGD in the logistic-regression notebook and Adam in the neural-network notebook).
6. Moving models and batches to CUDA when available.
7. Training with an explicit epoch/batch loop, backpropagation (`loss.backward()`) and optimizer updates.
8. Switching to evaluation mode and using `torch.no_grad()` for inference and test evaluation.
9. Tracking loss and classification accuracy, with visual inspection of results.

The later notebooks are organized around the named MLP, LeNet, AlexNet, VGGNet and ResNet model families. Check each notebook for its own architecture details, dataset, preprocessing, training configuration and evaluation procedure.

## Technical concepts and tools

- **PyTorch fundamentals:** tensors, tensor reshaping, device placement, `nn.Module`, layers and model forward methods.
- **Classification:** linear classification, multi-layer perceptrons, class logits and cross-entropy loss.
- **Computer vision data pipelines:** Fashion-MNIST, Torchvision datasets and transforms, batching and shuffling with DataLoader.
- **Optimization and autograd:** SGD, Adam, zeroing gradients, backpropagation and parameter updates.
- **Training and evaluation:** epochs, mini-batches, train/test splits, evaluation mode, gradient-free inference, loss and accuracy tracking.
- **CNN architecture study:** LeNet, AlexNet, VGGNet and ResNet.
- **Supporting tools:** Python, Matplotlib, Torch and Torchvision.

## Getting started

Open a notebook in [Google Colab](https://colab.research.google.com/) or a local Jupyter environment. The first two notebooks download Fashion-MNIST using Torchvision. Install compatible versions of PyTorch, Torchvision, Jupyter and Matplotlib if they are not already available.

Typical imports used in the introductory notebooks:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```

A CUDA-capable GPU is optional for the introductory Fashion-MNIST examples; the notebooks select CUDA when it is available and otherwise use the CPU.

## Learning path

Start with logistic regression to understand the classification pipeline and baseline model. Continue to the feed-forward neural network and advanced MLP, then study how convolutional architectures evolve through LeNet, AlexNet and VGGNet. Finish with ResNet to explore residual connections and deeper network design.

## Author

**Ali Hassan Abbas**  
[GitHub Profile](https://github.com/AliHassanAbbas) · [Repository](https://github.com/AliHassanAbbas/Pytorch-Projects-)
