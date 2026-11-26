# Linear Regression with PyTorch

A beginner-friendly implementation of **linear regression for exam-score prediction**, using PyTorch's `nn.Module` and `nn.Linear`. The notebook walks through the complete supervised-learning workflow—from preparing and visualizing a small dataset to training, evaluating, making a prediction for a new input, and saving model parameters.

## Notebook

[Open 0_Basic_Linear_Regression.ipynb](https://github.com/AliHassanAbbas/Pytorch-Projects-/blob/main/0%20Linear%20Regression/0_Basic_Linear_Regression.ipynb)

## Problem

The example uses hours studied as the input feature and exam score as the target. A small set of paired observations is used to learn a linear relationship and fit a prediction line.

## Model

The model uses a single linear layer:

```python
nn.Linear(in_features=1, out_features=1)
```

Conceptually, it learns:

`y = wx + b`

where `w` is the learnable weight and `b` is the bias.

## Workflow covered

1. Import PyTorch, NumPy and Matplotlib
2. Prepare the input and target arrays
3. Visualize the training data
4. Convert NumPy arrays to PyTorch tensors
5. Build a model using `nn.Module` and `nn.Linear`
6. Configure mean squared error (`nn.MSELoss`) and SGD
7. Train with a manual loop using `zero_grad()`, `backward()`, and `step()`
8. Evaluate using `model.eval()` and `torch.no_grad()`
9. Plot the fitted regression line
10. Predict the exam score for a new input (8 hours studied)
11. Save the learned parameters using `state_dict()` and `torch.save()`

## Training configuration

- Optimizer: Stochastic Gradient Descent (SGD)
- Learning rate: `1e-4`
- Loss: Mean Squared Error (MSE)
- Training epochs: 1,000
- Input/output features: 1/1

## Run

Open the notebook in Jupyter Notebook or Google Colab and execute cells in order. Requirements include PyTorch, NumPy and Matplotlib.

## Skills demonstrated

- Supervised learning and linear regression
- PyTorch tensors and NumPy-to-tensor conversion
- Model construction with `nn.Module` and `nn.Linear`
- Learnable weights and bias
- Forward pass and prediction
- MSE loss and SGD optimization
- Manual training loops and gradient-based learning
- Train/evaluation modes and inference without gradient tracking
- Plotting data, learned regression lines and predictions
- Saving model parameters with `state_dict`

## Scope

This is a simple educational regression example using a small dataset. It illustrates the mechanics of a PyTorch training workflow; it is not intended as a statistically validated exam-score prediction model.

## Author

Ali Hassan Abbas | [GitHub](https://github.com/AliHassanAbbas)
