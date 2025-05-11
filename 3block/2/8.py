import torch.nn as nn

def create_model():
    model = nn.Sequential(
        nn.Linear(100, 10),   # Входной слой: 100 → 10
        nn.ReLU(),            # Нелинейность
        nn.Linear(10, 1)      # Выходной слой: 10 → 1
    )
    return model