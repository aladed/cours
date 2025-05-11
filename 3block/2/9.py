import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

def train(model: Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()  # Переводим модель в режим обучения
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in data_loader:
        optimizer.zero_grad()                  # 1. Зануление градиентов
        outputs = model(X_batch)               # 2. Прямой проход
        loss = loss_fn(outputs, y_batch)       # 3. Вычисление ошибки
        loss.backward()                        # 4. Обратный проход
        optimizer.step()                       # 5. Шаг оптимизации

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss