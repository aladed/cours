import torch

def function03(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Инициализируем веса (используем равномерное распределение и разрешаем градиенты)
    weights = torch.rand(
        x.size(1),  # размерность вектора весов = число признаков
        dtype=torch.float32,
        requires_grad=True
    )
    
    # Скорость обучения (learning rate)
    learning_rate = 1e-2
    
    # Цикл градиентного спуска
    for _ in range(1000):  # эпохи (можно увеличить, если MSE не сходится)
        # Предсказание: y_pred = X @ w
        y_pred = x @ weights
        
        # Вычисляем MSE: mean((y_pred - y)^2)
        mse = torch.mean((y_pred - y) ** 2)
        
        # Обратное распространение (вычисление градиентов)
        mse.backward()
        
        # Обновляем веса в сторону антиградиента (градиентный спуск)
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            
            # Зануляем градиенты для следующей итерации
            weights.grad.zero_()
        
        # Ранняя остановка, если MSE < 1.0
        if mse.item() < 1.0:
            break
    
    # Отключаем градиенты перед возвратом весов
    return weights.detach()