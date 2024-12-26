import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv', sep=';', header=0)
data['Дата'] = pd.to_datetime(data['Дата'], format='%d.%m.%Y')
data.set_index('Дата', inplace=True)

# Пересэмплирование данных с использованием 'ME' (конец месяца)
monthly_prices = data.resample('ME').mean()

# Вычисление ежемесячной доходности
monthly_returns = monthly_prices.pct_change().dropna()

# Вычисление ожидаемой годовой доходности (средняя ежемесячная доходность * 12)
annual_returns = monthly_returns.mean() * 12

# Вычисление ковариационной матрицы доходностей (аннуализированной)
cov_matrix = monthly_returns.cov() * 12

cov_matrix_inverse = np.linalg.inv(cov_matrix)

# Вывод результатов
print("Среднегодовые доходности:")
print(annual_returns)
print("\nКовариационная матрица:")
print(cov_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, 
            xticklabels=cov_matrix.columns, yticklabels=cov_matrix.columns)
plt.title('Ковариационная матрица')
plt.show()
print("\nКовариационная матрица в -1:")
print(cov_matrix_inverse)

# Создаем вектор e из единиц (размерности равной числу активов)
e = np.ones(len(annual_returns))

# Вычисляем числитель: Σ^(-1) * e
numerator = np.dot(cov_matrix_inverse, e)

# Вычисляем знаменатель: e^T * Σ^(-1) * e
denominator = np.dot(e.T, np.dot(cov_matrix_inverse, e))


# Рассчитываем X*_min
X_min = numerator / denominator

print("\nX_min:")
# Вывод результата
print(X_min)

# Вычисляем Z* по формуле
mu = annual_returns  # Ожидаемая доходность активов

# Числитель: Σ^(-1) * μ
numerator_Z = np.dot(cov_matrix_inverse, mu)

# Знаменатель: e^T * Σ^(-1) * μ
denominator_Z = np.dot(e.T, np.dot(cov_matrix_inverse, mu))

# Σ^(-1) * e
inv_cov_e = np.dot(cov_matrix_inverse, e)

# Рассчитываем Z*
Z_star = numerator_Z - (denominator_Z / np.dot(e.T, np.dot(cov_matrix_inverse, e))) * inv_cov_e
print(np.sum(Z_star))

# Выводим результат
print("\nZ_star:")
print(Z_star)

import numpy as np
import pandas as pd

# Создание списка значений τ от 0 до 3 с шагом 0.1
tau_values = np.arange(0, 3.1, 0.1)

# Список для хранения результатов
X_star_values = []

# Вычисление X* для каждого τ
for tau in tau_values:
    X_star = X_min + tau * Z_star
    X_star_values.append(X_star)

# Создание таблицы для τ и X_i
X_star_table = pd.DataFrame(X_star_values, columns=[f'X_{i+1}' for i in range(len(X_min))], index=tau_values)

# Вывод таблицы
print("\nX_star_table:")
print(X_star_table)

# Вычисляем μ_X* и σ_X* для каждого τ
mu_X_star_values = []
sigma_X_star_values = []

for tau in tau_values:
    X_star = X_min + tau * Z_star
    mu_X_star = np.dot(X_star, annual_returns)  # Ожидаемая доходность
    sigma_X_star = np.sqrt(np.dot(X_star.T, np.dot(cov_matrix, X_star)))  # Риск (стандартное отклонение)

    mu_X_star_values.append(mu_X_star)
    sigma_X_star_values.append(sigma_X_star)

print("\nmu_X_star_values:")
print(mu_X_star_values)

import matplotlib.pyplot as plt

# Построение точечного графика для μ_X* и σ_X* от τ
plt.figure(figsize=(10, 6))

# Строим точечный график
plt.scatter(mu_X_star_values, sigma_X_star_values, color='g', label='Портфели')

# Настройка графика
plt.xlabel('μ_X* (Ожидаемая доходность)')
plt.ylabel('σ_X* (Риск / Стандартное отклонение)')
plt.title('Зависимость μ_X* и σ_X*')
plt.grid(True)

# Отображаем график
plt.show()


# Вычисление дисперсии (σ²) для каждого τ
sigma_squared_values = [sigma**2 for sigma in sigma_X_star_values]

# Построение графика μ и σ²
plt.figure(figsize=(10, 6))

# Строим точечный график
plt.scatter(mu_X_star_values, sigma_squared_values, color='g', label='Портфели')

# Настройка графика
plt.xlabel('μ (Ожидаемая доходность)')
plt.ylabel('σ² (Дисперсия)')
plt.title('Зависимость μ и σ²')
plt.grid(True)

# Отображаем график
plt.legend()
plt.show()