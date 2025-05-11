from torchvision.datasets import MNIST
import torchvision.transforms as T
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
from time import perf_counter
from IPython.display import clear_output
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import nn


def count_parameters_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool):
    return ((in_channels*((kernel_size)**2)) + bias) * out_channels



mnist_train = MNIST("../datasets/mnist", train = True, download = True)
mnist_train = MNIST("../datasets/mnist", train = True, download = True, transform = T.ToTensor() )
mnist_valid = MNIST("../datasets/mnist", train = False, download = True, transform = T.ToTensor() )

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
valid_loader = DataLoader(mnist_valid, batch_size=64, shuffle=True)

def train(
    model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn
) -> tuple[float, float]:
    model.train()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(data_loader, desc='Train'):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()
    return total_loss / len(data_loader), correct / total

def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn) -> tuple[float, float]:
    model.eval()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(data_loader, desc='Evaluate'):
        output = model(x)
        loss = loss_fn(output, y)
        total_loss += loss.item()
        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()
    return total_loss / len(data_loader), correct / total

def plot_stats(train_loss, valid_loss, valid_accuracy, title):
    plt.figure(figsize=(16, 8))
    plt.title(title + ' loss')
    plt.plot(train_loss, label = 'Train loss')
    plt.plot(valid_loss, label = 'Valid loss')
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(16, 8))
    plt.title(title + ' accuracy')
    plt.plot(valid_accuracy)
    plt.grid()


def create_mlp_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return model
model = create_mlp_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

epochs = 15

train_loss_history, valid_loss_history = [], []
valid_accuracy_histiry = []

start = perf_counter()

for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, loss_fn)

    valid_loss, valid_accuracy = evaluate(model, valid_loader, loss_fn)
    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)
    valid_accuracy_histiry.append(valid_accuracy)

    clear_output()

    plot_stats(train_loss_history, valid_loss_history, valid_accuracy_histiry, 'MLP model')

print(f'Total time {perf_counter() - start:.5f}')


# Получаем state_dict
state_dict = model.state_dict()

print(state_dict.keys())  # Выведет: odict_keys(['0.weight', '0.bias', '2.weight', '2.bias'])
# Сохраняем только параметры модели (рекомендуемый способ)
state_dict = model.state_dict()
torch.save(model.state_dict(), 'model_weights.pth')






