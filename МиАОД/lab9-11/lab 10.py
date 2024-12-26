# Использование метрик, устойчивых к выбросам.
# Используйте набор данных diabetes из sklearn. и обучите модель HuberRegressor, которая менее чувствительна к выбросам по сравнению с MSE-метрикой. 
# Сравните эту модель с базовой моделью линейной регрессии.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Шаг 1: Загрузка набора данных
data = load_diabetes()
X = data.data
y = data.target

# Шаг 2: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Обучение базовой модели линейной регрессии
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Прогнозирование и оценка модели
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

# Шаг 4: Обучение модели HuberRegressor
huber_model = HuberRegressor()
huber_model.fit(X_train, y_train)

# Прогнозирование и оценка модели HuberRegressor
y_pred_huber = huber_model.predict(X_test)
mse_huber = mean_squared_error(y_test, y_pred_huber)
mae_huber = mean_absolute_error(y_test, y_pred_huber)

# Шаг 5: Сравнение результатов
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Huber Regressor'],
    'MSE': [mse_linear, mse_huber],
    'MAE': [mae_linear, mae_huber]
})

print(results)

# Шаг 6: Визуализация результатов
plt.bar(results['Model'], results['MSE'], color=['blue', 'orange'], alpha=0.6, label='MSE')
plt.bar(results['Model'], results['MAE'], color=['lightblue', 'lightcoral'], alpha=0.6, label='MAE')
plt.ylabel('Error')
plt.title('Comparison of Linear Regression and Huber Regressor')
plt.legend()
plt.show()