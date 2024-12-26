import numpy as np

# Метод Рунге-Кутты для построения начальных значений
def runge_kutta(f, x0, y0, h, n):
    x_values = [x0]
    y_values = [y0]

    for i in range(n):
        k1 = f(x_values[-1], y_values[-1])
        k2 = f(x_values[-1] + h / 2, y_values[-1] + h / 2 * k1)
        k3 = f(x_values[-1] + h / 2, y_values[-1] + h / 2 * k2)
        k4 = f(x_values[-1] + h, y_values[-1] + h * k3)

        x_values.append(x_values[-1] + h)
        y_values.append(y_values[-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    return x_values, y_values


# Функция правой части дифференциального уравнения для метода Рунге-Кутты
def f(x, u):
    return (u + np.sqrt(x**2 + u**2)) / x


# Метод простой итерации для интерполяционного метода Адамса 3-го порядка
def adams_method(x_values, y_values, h):
    """
    x_values: значения x
    y_values: значения y, соответствующие x_values
    h: шаг
    """
    n = len(x_values)

    # Инициализация значений для метода Адамса (по методу Рунге-Кутты)
    u_values = y_values

    # Вычисление остальных значений методом простой итерации
    for i in range(n - 1, 2, -1):
        # Начальное приближение для u_{i+1}
        u_next = u_values[i]

        # Метод простой итерации
        max_iterations = 100
        epsilon = 1e-7
        for _ in range(max_iterations):
            u_next_old = u_next
            u_next = y_values[i] + h / 12 * (
                        5 * f(x_values[i], u_next) + 8 * f(x_values[i - 1], u_values[i - 1]) - f(x_values[i - 2],
                                                                                                 u_values[i - 2]))
            if abs(u_next - u_next_old) < epsilon:
                break

        u_values.append(u_next)

    return u_values


# Начальные условия
x0 = 1
y0 = 0
h = 0.05
n = 10

# Построение начальных значений методом Рунге-Кутты
x_values, y_values = runge_kutta(f, x0, y0, h, n)

# Вычисление остальных значений методом Адамса
u_values = adams_method(x_values, y_values, h)

print("Метод Адамса:")
for i in range(len(x_values)):
    print(f" u({x_values[i]:.2f}) = {u_values[i]:.8f}")