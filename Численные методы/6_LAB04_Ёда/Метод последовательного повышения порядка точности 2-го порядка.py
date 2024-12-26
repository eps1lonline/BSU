import numpy as np

def f(x, u):
    return (u + np.sqrt(x**2 + u**2)) / x

u0 = 0
n = 10
a = 1
b = 1.5

def increase_accuracy(a, b, n, f, y0):
    h = (b - a) / n
    x = [a + h * i for i in range(n + 1)]
    y = [y0]

    for j in range(n):
        y_half = y[j] + h / 2 * f(x[j], y[j])
        y.append(y[j] + h * f(x[j] + 1 / 2 * h, y_half))

    return x, y

sol = increase_accuracy(a, b, n, f, u0)
print('Метод последовательного повышения порядка точности')

for x, y in zip(sol[0], sol[1]):
    print(f'u({x:.2f}) = {y:.8f}')
