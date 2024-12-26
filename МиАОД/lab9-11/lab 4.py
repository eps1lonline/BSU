# Построение Lasso регрессии.
# Также примените Lasso регрессию к тем же данным. При подборе гиперпараметра alpha через кросс-валидацию сравните количество нулевых весов в модели с результатами Ridge регрессии.

# это метод линейной регрессии, который включает в себя регуляризацию с использованием L1 нормы.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

# Загружаем данные
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Определяем модель Ridge регрессии для сравнения
ridge = Ridge()
param_grid_ridge = {'alpha': np.logspace(-4, 4, 100)}
grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid_ridge,
                                  scoring='neg_mean_squared_error', cv=10)
grid_search_ridge.fit(X, y)

# Определяем модель Lasso регрессии
lasso = Lasso(max_iter=10000)
param_grid_lasso = {'alpha': np.logspace(-4, 4, 100)}
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid_lasso,
                                  scoring='neg_mean_squared_error', cv=10)
grid_search_lasso.fit(X, y)

# Результаты Ridge
best_alpha_ridge = grid_search_ridge.best_params_['alpha']
best_mse_ridge = -grid_search_ridge.best_score_

# Результаты Lasso
best_alpha_lasso = grid_search_lasso.best_params_['alpha']
best_mse_lasso = -grid_search_lasso.best_score_
lasso_model = grid_search_lasso.best_estimator_

# Количество нулевых весов в Lasso
num_zero_weights_lasso = np.sum(lasso_model.coef_ == 0)

# Выводим результаты
print(f'Лучший гиперпараметр alpha для Ridge: {best_alpha_ridge}')
print(f'Лучшее среднее значение MSE для Ridge: {best_mse_ridge}')
print(f'Лучший гиперпараметр alpha для Lasso: {best_alpha_lasso}')
print(f'Лучшее среднее значение MSE для Lasso: {best_mse_lasso}')
print(f'Количество нулевых весов в Lasso: {num_zero_weights_lasso}')