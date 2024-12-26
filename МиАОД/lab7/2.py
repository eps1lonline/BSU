# Задание 2: Исключение мультиколлинеарных признаков.
# Используйте набор данных "Wine" из sklearn.datasets. 
# Вычислите корреляционную матрицу, а затем найдите и исключите признаки, у которых корреляция друг с другом превышает заданный порог.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Загрузка набора данных Wine
wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# Вычисление корреляционной матрицы
correlation_matrix = wine_df.corr()

# Визуализация корреляционной матрицы с помощью heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Корреляционная матрица')
plt.show()

# Установка порога для корреляции
threshold = 0.8

# Находим признаки, которые нужно исключить
def get_redundant_pairs(correlation_matrix):
    pairs_to_drop = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                pairs_to_drop.add(correlation_matrix.columns[i])
    return pairs_to_drop

# Получаем признаки для исключения
redundant_features = get_redundant_pairs(correlation_matrix)

# Исключаем мультиколлинеарные признаки
wine_df_reduced = wine_df.drop(columns=redundant_features)

# Выводим результаты
print("Исключенные признаки:", redundant_features)
print("Размерность оригинального набора данных:", wine_df.shape)
print("Размерность уменьшенного набора данных:", wine_df_reduced.shape)