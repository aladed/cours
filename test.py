import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Счётчики для вычислений функции
function_calls = 0

# Определение функции с подсчётом вычислений
def f(x):
    global function_calls
    function_calls += 1
    return 5 * x**2 * np.exp(-x / 2)

# Поиск максимума методом Брента (через минимизацию -f(x))
result = minimize_scalar(
    lambda x: -f(x),  # Минимизируем -f(x), чтобы найти максимум f(x)
    bracket=(2, 6),   # Интервал поиска
    method='brent'    # Метод Брента
)

x_max = result.x
y_max = f(x_max)  # Учитываем последнее вычисление функции

# Оценка количества итераций
# Метод Брента вызывает функцию примерно 2-3 раза за итерацию
iteration_count = function_calls // 2

# Вывод результата
print(f"Максимум функции находится в точке x = {x_max:.4f}, y = {y_max:.4f}")
print(f"Количество итераций: {iteration_count}")
print(f"Количество вычислений функции: {function_calls}")
