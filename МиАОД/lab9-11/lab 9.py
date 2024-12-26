# Комбинирование методов отбора признаков и регуляризации.
# Выберите подмножество признаков с помощью любого метода отбора признаков, а затем обучите модели с Lasso и Ridge регуляризацией. 
# Сравнивай модели между собой и с моделью, построенной на всех признаках.

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression

# Шаг 1: Загрузка набора данных
data = load_diabetes()
X = data.data
y = data.target

# Шаг 2: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Применение метода отбора признаков
selector = SelectKBest(score_func=f_regression, k=5)  # Выбираем 5 лучших признаков
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Шаг 4: Обучение моделей на всех признаках
lasso_all = Lasso(alpha=1.0)
ridge_all = Ridge(alpha=1.0)

lasso_all.fit(X_train, y_train)
ridge_all.fit(X_train, y_train)

# Прогнозирование и оценка моделей на всех признаках
y_pred_lasso_all = lasso_all.predict(X_test)
y_pred_ridge_all = ridge_all.predict(X_test)

mse_lasso_all = mean_squared_error(y_test, y_pred_lasso_all)
mse_ridge_all = mean_squared_error(y_test, y_pred_ridge_all)

# Шаг 5: Обучение моделей на отобранных признаках
lasso_selected = Lasso(alpha=1.0)
ridge_selected = Ridge(alpha=1.0)

lasso_selected.fit(X_train_selected, y_train)
ridge_selected.fit(X_train_selected, y_train)

# Прогнозирование и оценка моделей на отобранных признаках
y_pred_lasso_selected = lasso_selected.predict(X_test_selected)
y_pred_ridge_selected = ridge_selected.predict(X_test_selected)

mse_lasso_selected = mean_squared_error(y_test, y_pred_lasso_selected)
mse_ridge_selected = mean_squared_error(y_test, y_pred_ridge_selected)

# Шаг 6: Сравнение результатов
results = pd.DataFrame({
    'Model': ['Lasso (All Features)', 'Ridge (All Features)', 'Lasso (Selected Features)', 'Ridge (Selected Features)'],
    'MSE': [mse_lasso_all, mse_ridge_all, mse_lasso_selected, mse_ridge_selected]
})

print(results)

# Вывод коэффициентов для анализа
print("\nLasso Coefficients (All Features):", lasso_all.coef_)
print("Lasso Coefficients (Selected Features):", lasso_selected.coef_)
print("\nRidge Coefficients (All Features):", ridge_all.coef_)
print("Ridge Coefficients (Selected Features):", ridge_selected.coef_)