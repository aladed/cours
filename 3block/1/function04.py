import torch
import torch.nn as nn

def function04(x: torch.Tensor, y: torch.Tensor):
    # Приведение y к форме (n, 1) для корректной работы с nn.Linear
    y = y.view(-1, 1)

    # Полносвязный слой: число входов = число признаков, выходов = 1
    model = nn.Linear(x.size(1), 1)

    # Функция потерь: среднеквадратичная ошибка
    criterion = nn.MSELoss()
    # Оптимизатор: стохастический градиентный спуск
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Обучение
    for epoch in range(1000):
        # Прямой проход
        y_pred = model(x)

        # Вычисление ошибки
        loss = criterion(y_pred, y)

        # Обратное распространение ошибки
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Ранний выход, если достигли нужного качества
        if loss.item() < 0.3:
            break
        

    return model
