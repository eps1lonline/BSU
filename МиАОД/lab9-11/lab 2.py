# Использование кросс-валидации.
# С использованием того же набора данных проведите k-fold кросс-валидацию (k=10) для своей модели и сравните среднее значения MSE на всех фолдах.

# разделить набор данных на несколько частей и использовать их для обучения и тестирования модели.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# Загружаем данные
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Создаем модель линейной регрессии
model = LinearRegression()

# Задаем k-fold кросс-валидацию с k=10
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Выполняем кросс-валидацию и вычисляем MSE
mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

# Выводим результаты
print(f'MSE для каждого фолда: {mse_scores}')
print(f'Среднее значение MSE: {np.mean(mse_scores)}')