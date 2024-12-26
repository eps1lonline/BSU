import numpy as np
import matplotlib.pyplot as plt

# Задаем параметры
L = 1.0  # Длина области
T = 1.0  # Временной интервал
Nx = 10  # Число узлов по пространству
Nt = 1000  # Число шагов по времени (увеличено для выполнения условия Куранта)

dx = L / Nx  # Шаг по пространству
dt = T / Nt  # Шаг по времени
x = np.linspace(0, L, Nx + 1)
t = np.linspace(0, T, Nt + 1)

# Проверяем условие Куранта
assert dt / dx**2 <= 0.5, "Нарушено условие устойчивости Куранта!"

# Определяем параметры для σ_α и σ*
alpha = 0.5  # Параметр для σ_α
sigma_alpha = 1 / 2 + alpha * (dx**2 / dt)  # σ_α
sigma_star = 1 / 2 - (dx**2 / (12 * dt))  # σ*

# Значения sigma для исследования
sigma_values = [0, 1, 0.5, sigma_alpha, sigma_star]

# Вывод значений σ_α и σ*
print(f"σ_alpha = {sigma_alpha:.4f}")
print(f"σ* = {sigma_star:.4f}")

# Функция начального условия
def initial_condition(x):
    return np.cos(np.pi * x / 2)

# Правая часть уравнения
def f(x, t):
    return np.cos(2 * (x + t)) + np.sqrt(2) * np.cos(np.pi / 4 + x + t)

# Коэффициент перед производной
def a(x, t):
    return np.cos(x + t) + 1

# Граничные условия
def boundary_conditions(t):
    return np.cos(t), -np.sin(t + 1)

# Решение методом весов
def solve_sigma(sigma):
    u = np.zeros((Nt + 1, Nx + 1))  # Матрица решения
    u[0, :] = initial_condition(x)  # Начальное условие

    for n in range(0, Nt):
        # Применение граничных условий
        u[n + 1, 0], u[n + 1, -1] = boundary_conditions(t[n + 1])

        # Построение системы уравнений
        A = np.zeros((Nx - 1, Nx - 1))
        B = np.zeros((Nx - 1))

        for i in range(1, Nx):
            a_mid = a(x[i], t[n])  # Коэффициент перед производной
            if i > 1:
                A[i - 1, i - 2] = -sigma * dt * a_mid / dx**2
            A[i - 1, i - 1] = 1 + 2 * sigma * dt * a_mid / dx**2
            if i < Nx - 1:
                A[i - 1, i] = -sigma * dt * a_mid / dx**2

            B[i - 1] = (
                u[n, i]
                + (1 - sigma) * dt * a_mid * (u[n, i - 1] - 2 * u[n, i] + u[n, i + 1]) / dx**2
                + dt * f(x[i], t[n])
            )

        # Решение системы уравнений
        u_next = np.linalg.solve(A, B)
        u[n + 1, 1:-1] = u_next

    return u

# Построим и отобразим решения для разных sigma
plt.figure(figsize=(10, 6))

for sigma in sigma_values:
    u = solve_sigma(sigma)
    plt.plot(x, u[-1, :], label=f"σ = {sigma:.4f}")

plt.title("Сравнение решений для второго уравнения при разных σ")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.grid()
plt.show()