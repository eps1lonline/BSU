import numpy as np

def chebyshev_method(func, x0, epsilon=1e-7, max_iter=1000):
    x_prev = x0
    x_curr = x0
    iteration = 0

    while True:
        iteration += 1
        x_next = x_curr - func(x_curr) / chebyshev_derivative(func, x_curr)

        if np.abs(x_next - x_curr) <= epsilon or iteration >= max_iter:
            break

        x_prev = x_curr
        x_curr = x_next

    return x_next, iteration

def func(x):
    return x**3 + 3*x + 1

def chebyshev_derivative(func, x):
    h = 1e-5
    return (func(x + h / 2) - func(x - h / 2)) / h

def residual(func, root):
    return np.abs(func(root))

x0 = -0.25
root, iterations = chebyshev_method(func, x0)
residual_value = residual(func, root)

print("Корень уравнения:", root)
print("Невязка:", residual_value)
print("Количество итераций:", iterations)