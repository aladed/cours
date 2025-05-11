import torch
from torch.nn import Module
from torch.utils.data import DataLoader

def evaluate(model: Module, data_loader: DataLoader, loss_fn):
    model.eval()  # Переводим модель в режим инференса
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Отключаем вычисление градиентов
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)               # Прямой проход
            loss = loss_fn(outputs, y_batch)       # Вычисление ошибки
            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss