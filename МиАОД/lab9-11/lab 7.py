# Изучение влияния регуляризации.
# Выберите набор данных с высокой размерностью признаков. Создайте модели Lasso и Ridge регрессии. Проведите эксперименты с различными степенями 
# регуляризации и установите, как они влияют на производительность модели и распределение весов признаков.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

# Шаг 1: Загрузка набора данных
data = load_diabetes()
X = data.data
y = data.target

# Добавим дополнительные синтетические признаки
np.random.seed(42)
X = np.hstack((X, np.random.randn(X.shape[0], 10)))  # Добавляем 10 случайных признаков

# Шаг 2: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Определение степеней регуляризации
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

# Шаг 4: Обучение моделей и оценка производительности
results = []

for alpha in alphas:
    # Lasso
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    
    # Ridge
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    
    results.append({
        'alpha': alpha,
        'Lasso MSE': lasso_mse,
        'Ridge MSE': ridge_mse,
        'Lasso Coefficients': lasso_model.coef_,
        'Ridge Coefficients': ridge_model.coef_
    })

# Преобразование результатов в DataFrame
results_df = pd.DataFrame(results)

# Шаг 5: Визуализация производительности
plt.figure(figsize=(12, 6))
plt.plot(results_df['alpha'], results_df['Lasso MSE'], marker='o', label='Lasso MSE', color='blue')
plt.plot(results_df['alpha'], results_df['Ridge MSE'], marker='o', label='Ridge MSE', color='red')
plt.xscale('log')
plt.xlabel('Regularization Strength (alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Regularization on Model Performance')
plt.legend()
plt.grid()
plt.show()

# Шаг 6: Анализ распределения весов признаков
lasso_coeffs = np.array([result['Lasso Coefficients'] for result in results])
ridge_coeffs = np.array([result['Ridge Coefficients'] for result in results])

plt.figure(figsize=(14, 7))
for i in range(lasso_coeffs.shape[1]):
    plt.plot(results_df['alpha'], lasso_coeffs[:, i], label=f'Lasso Coeff {i}')
for i in range(ridge_coeffs.shape[1]):
    plt.plot(results_df['alpha'], ridge_coeffs[:, i], linestyle='--', label=f'Ridge Coeff {i}')
plt.xscale('log')
plt.xlabel('Regularization Strength (alpha)')
plt.ylabel('Coefficient Value')
plt.title('Effect of Regularization on Coefficients')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.grid()
plt.show()