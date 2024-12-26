# Задание 3: Выбор наиболее значимых признаков.
# Используйте набор данных "Boston Housing" из sklearn.datasets. 
# Вычислите коэффициенты корреляции между каждым признаком и целевой переменной, затем выберите n признаков с наибольшим абсолютным значением коэффициента.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Получение набора данных о жилье в Бостоне
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Преобразование в DataFrame
boston_df = pd.DataFrame(data, columns=[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
])
boston_df['MEDV'] = target  # Добавляем целевую переменную

# Вычисление коэффициентов корреляции
correlation_matrix = boston_df.corr()
correlation_with_target = correlation_matrix['MEDV'].drop('MEDV')

# Визуализация корреляционной матрицы с помощью heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Корреляционная матрица')
plt.show()

# Выбор n признаков с наибольшим абсолютным значением коэффициента
n = 5  # Задайте количество признаков для выбора
top_n_features = correlation_with_target.abs().nlargest(n).index

# Вывод результатов
print("Наиболее значимые признаки:")
print(top_n_features)