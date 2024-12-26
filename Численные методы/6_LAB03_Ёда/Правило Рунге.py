import numpy as np

def f(x):
    return x*(1+x)**(1/3)

def composite_right_rectangle(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return h * np.sum(f(x))

def runge_rule(f, a, b, epsilon):
    n = 10
    I_n = composite_right_rectangle(f, a, b, n)

    while True:
        n *= 2
        I_2n = composite_right_rectangle(f, a, b, n)

        if np.abs(I_2n - I_n) <= epsilon:
            break

        I_n = I_2n
    h = (b - a) / n

    return I_2n, n, h

a = 1
b = 9
epsilon = 1e-5

integral, n, h = runge_rule(f, a, b, epsilon)

print(f"Интеграл: {integral:.6f}")
print(f"Количество разбиений: {n}")
print(f"Шаг для разбиения: {h:.6f}")