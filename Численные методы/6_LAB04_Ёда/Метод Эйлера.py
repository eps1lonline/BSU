import numpy as np

def f(x, u):
    return (u + np.sqrt(x**2 + u**2)) / x

u0 = 0
n = 10
a = 1
b = 1.5

def euler(a, b, n, f, y0):
    h = (b - a) / n
    x = [a + h*i for i in range(n+1)]
    y = [y0]

    for i in range(n):
        y.append(y[i] + h*f(x[i], y[i]))

    return x, y

sol = euler(a, b, n, f, u0)
print('Метод Эйлера')

for x, y in zip(sol[0], sol[1]): print(f'u({x:.2f}) = {y:.8f}')