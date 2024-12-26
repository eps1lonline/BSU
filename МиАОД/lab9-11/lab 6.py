# Исследование эффекта масштабирования признаков.
# Используйте любой набор данных с числовыми признаками. Тренируйте модели Ridge и Lasso регрессии на исходных данных и 
# предобработанных данных (используйте стандартизацию и нормализацию). Сравните коэффициенты моделей, полученных для исходных и предобработанных данных.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Шаг 1: Загрузка набора данных
data = fetch_california_housing()
X = data.data
y = data.target

# Шаг 2: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Обучение моделей на исходных данных
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=1.0)

ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Коэффициенты для исходных данных
ridge_coeffs_original = ridge_model.coef_
lasso_coeffs_original = lasso_model.coef_

# Шаг 4: Стандартизация и нормализация данных
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# Шаг 5: Обучение моделей на стандартизированных данных
ridge_model_standard = Ridge(alpha=1.0)
lasso_model_standard = Lasso(alpha=1.0)

ridge_model_standard.fit(X_train_standard, y_train)
lasso_model_standard.fit(X_train_standard, y_train)

# Коэффициенты для стандартизированных данных
ridge_coeffs_standard = ridge_model_standard.coef_
lasso_coeffs_standard = lasso_model_standard.coef_

# Шаг 6: Обучение моделей на нормализованных данных
ridge_model_minmax = Ridge(alpha=1.0)
lasso_model_minmax = Lasso(alpha=1.0)

ridge_model_minmax.fit(X_train_minmax, y_train)
lasso_model_minmax.fit(X_train_minmax, y_train)

# Коэффициенты для нормализованных данных
ridge_coeffs_minmax = ridge_model_minmax.coef_
lasso_coeffs_minmax = lasso_model_minmax.coef_

# Шаг 7: Сравнение коэффициентов моделей
coefficients = pd.DataFrame({
    'Feature': data.feature_names,
    'Ridge (Original)': ridge_coeffs_original,
    'Lasso (Original)': lasso_coeffs_original,
    'Ridge (Standardized)': ridge_coeffs_standard,
    'Lasso (Standardized)': lasso_coeffs_standard,
    'Ridge (Normalized)': ridge_coeffs_minmax,
    'Lasso (Normalized)': lasso_coeffs_minmax
})

print(coefficients)

# Визуализация коэффициентов
coefficients.set_index('Feature').plot(kind='bar', figsize=(14, 7), title='Comparison of Coefficients')
plt.axhline(0, color='grey', lw=0.8)
plt.show()