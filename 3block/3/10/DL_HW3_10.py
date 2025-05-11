from torchvision.datasets import MNIST
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from time import perf_counter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Optimizer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используется устройство:", device)

def count_parameters_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool):
    return ((in_channels * (kernel_size ** 2)) + bias) * out_channels



mnist_train = MNIST("../datasets/mnist", train=True, download=True, transform=T.ToTensor())
mnist_valid = MNIST("../datasets/mnist", train=False, download=True, transform=T.ToTensor())

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, pin_memory=True)
valid_loader = DataLoader(mnist_valid, batch_size=64, shuffle=False, pin_memory=True)



def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn) -> tuple[float, float]:
    model.train()
    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(data_loader, desc='Train'):
        x, y = x.to(device), y.to(device)
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

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc='Evaluate'):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            _, y_pred = torch.max(output, 1)
            total += y.size(0)
            correct += (y_pred == y).sum().item()
    return total_loss / len(data_loader), correct / total



def create_conv_model():
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(128 * 3 * 3, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return model



model = create_conv_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_loss_history, valid_loss_history = [], []
valid_accuracy_history = []

start = perf_counter()
max_epochs = 100

for epoch in range(max_epochs):
    train_loss, _ = train(model, train_loader, optimizer, loss_fn)
    valid_loss, valid_accuracy = evaluate(model, valid_loader, loss_fn)

    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)
    valid_accuracy_history.append(valid_accuracy)

    print(f"Epoch {epoch + 1}: valid accuracy = {valid_accuracy:.4f}")

    if valid_accuracy >= 0.993:
        print(f"Достигнута точность 99.3% на эпохе {epoch + 1}")
        torch.save(model.state_dict(), 'model_weights.pth')
        break

print(f'Total time {perf_counter() - start:.5f} сек.')