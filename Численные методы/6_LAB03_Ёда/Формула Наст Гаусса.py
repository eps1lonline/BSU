import numpy as np
import math

def f(x):
    return x*(1 + x)**1/3

# Узлы и веса для квадратурной формулы Гаусса при n=4
nodes = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
weights = np.array([0.34785484513745385, 0.6521451548625461, 0.6521451548625461, 0.34785484513745385])

# Преобразование переменной
def transform(t):
    return 8 * t + 10

# Ошибка
def residual_error(f_2n_2_eta):
    return 2 / ((2*4 + 3) * math.factorial(2*4 + 2)) * (math.factorial(4 + 1) / (2 * math.factorial(2*4 + 2))**2) * f_2n_2_eta

# Вычисление интеграла по формуле Гаусса
integral = 0.9 * sum(weights[i] * f(transform(nodes[i])) for i in range(4))

print(f"Интеграл: {integral:.6f}")

f_2n_2_eta = 4.92081 * 10**6
error = residual_error(f_2n_2_eta)

print(f"Оценка погрешности: {error:.6e}")