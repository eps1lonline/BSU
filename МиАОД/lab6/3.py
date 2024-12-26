# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Загрузка набора данных о жилье в Калифорнии
housing = fetch_california_housing(as_frame=True)
df_california = housing.frame

# Подготовка данных
X = df_california.drop(columns='MedHouseVal')  # Удаляем целевую переменную
y = df_california['MedHouseVal']  # Медицинская стоимость дома

# Добавление константы для OLS регрессии
X = sm.add_constant(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Подгонка модели OLS
model = sm.OLS(y_train, X_train).fit()

# Получение p-values для признаков
p_values = model.pvalues

# Вывод p-values
print("P-значения признаков (California Housing Dataset):")
print(p_values)

# Отбор признаков на основе p-value
alpha = 0.05  # уровень значимости
selected_features = p_values[p_values < alpha].index.tolist()

# Вывод выбранных признаков
print("\nВыбранные признаки на основе p-value (California Housing Dataset):")
print(selected_features)

# Обучение модели только на выбранных признаках
if 'const' in selected_features:
    selected_features.remove('const')  # Удаляем константу

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Обучение модели на выбранных признаках
model_selected = sm.OLS(y_train, X_train_selected).fit()

# Вывод результатов модели
print("\nРезультаты модели с выбранными признаками (California Housing Dataset):")
print(model_selected.summary())