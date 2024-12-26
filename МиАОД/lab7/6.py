# Задание 6: Исследование влияния предобработки данных на результаты корреляционного анализа.
# Примените различные методы предобработки (например, нормализацию, стандартизацию, логарифмирование) к данным перед вычислением корреляции и сравните полученные результаты.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Шаг 1: Загрузка данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Разделим данные на признаки и целевую переменную
X = data.drop('quality', axis=1)
y = data['quality']

# Функция для вычисления и визуализации корреляции
def compute_and_plot_correlation(X, title):
    spearman_corr = X.corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(title)
    plt.show()

# Шаг 2: Корреляция без предобработки
compute_and_plot_correlation(X, 'Корреляция Спирмена (без предобработки)')

# Шаг 3: Нормализация
scaler_minmax = MinMaxScaler()
X_normalized = pd.DataFrame(scaler_minmax.fit_transform(X), columns=X.columns)
compute_and_plot_correlation(X_normalized, 'Корреляция Спирмена (нормализация)')

# Шаг 4: Стандартизация
scaler_standard = StandardScaler()
X_standardized = pd.DataFrame(scaler_standard.fit_transform(X), columns=X.columns)
compute_and_plot_correlation(X_standardized, 'Корреляция Спирмена (стандартизация)')

# Шаг 5: Логарифмирование
X_log_transformed = np.log1p(X)  # Используем np.log1p для логарифмирования (x + 1)
compute_and_plot_correlation(X_log_transformed, 'Корреляция Спирмена (логарифмирование)')