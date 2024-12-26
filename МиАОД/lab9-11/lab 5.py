# Использование других функций потерь.
# Используйте набор данных diabetes из sklearn.datasets и постройте модель HuberRegressor - линейную модель с функцией потерь Хьюбера, которая менее чувствительна к выбросам по сравнению с MSE.
# Сравнение моделей
# HuberRegressor — это метод линейной регрессии, который сочетает в себе преимущества обычной линейной регрессии и регрессии с использованием метода наименьших абсолютных отклонений

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Шаг 1: Загрузка набора данных
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Шаг 2: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Обучение модели HuberRegressor
huber_regressor = HuberRegressor()
huber_regressor.fit(X_train, y_train)

# Шаг 4: Обучение модели LinearRegression для сравнения
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Шаг 5: Прогнозирование и оценка моделей
y_pred_huber = huber_regressor.predict(X_test)
y_pred_linear = linear_regressor.predict(X_test)

# Шаг 6: Оценка производительности
mse_huber = mean_squared_error(y_test, y_pred_huber)
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_huber = mean_absolute_error(y_test, y_pred_huber)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

# Вывод результатов
print(f'HuberRegressor MSE: {mse_huber}, MAE: {mae_huber}')
print(f'LinearRegression MSE: {mse_linear}, MAE: {mae_linear}')

# Шаг 7: Визуализация (по желанию)
plt.scatter(y_test, y_pred_huber, label='HuberRegressor predictions', color='blue')
plt.scatter(y_test, y_pred_linear, label='LinearRegression predictions', color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Comparison of HuberRegressor and LinearRegression')
plt.legend()
plt.show()