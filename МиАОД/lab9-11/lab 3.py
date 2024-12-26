# Построение Ridge регрессии.
# Примените Ridge регрессию к набору данных "Boston Housing". Подберите гиперпараметр alpha через кросс-валидацию. 

# Оптимизирует модифицированную функцию потерь, которая включает как ошибку предсказания, так и штраф за величину коэффициентов.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Загружаем данные
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Определяем модель Ridge регрессии
ridge = Ridge()

# Задаем параметры для подбора гиперпараметра alpha
param_grid = {'alpha': np.logspace(-4, 4, 100)}

# Настраиваем GridSearchCV для подбора alpha с 10-кратной кросс-валидацией
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=10)

# Обучаем модель
grid_search.fit(X, y)

# Выводим результаты
print(f'Лучший гиперпараметр alpha: {grid_search.best_params_["alpha"]}')
print(f'Лучшее среднее значение MSE: {-grid_search.best_score_}')