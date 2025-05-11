import torch

def function02(dataset: torch.Tensor) -> torch.Tensor:
    # Получаем количество признаков (столбцов) в датасете
    num_features = dataset.size(1)
    
    # Создаем тензор весов из равномерного распределения [0, 1]
    weights = torch.rand(
        num_features,  # размерность вектора весов
        dtype=torch.float32,  # тип float32
        requires_grad=True  # разрешаем вычисление градиентов
    )
    
    return weights
    