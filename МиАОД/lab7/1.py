# Задание 1: Работа с корреляционной матрицей.
# Используйте набор данных "Iris" из sklearn.datasets. 
# Вычислите корреляционную матрицу числовых признаков. 
# Затем визуализируйте эту матрицу с помощью heatmap в библиотеке seaborn.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Загрузка набора данных Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Вычисление корреляционной матрицы
correlation_matrix = iris_df.corr()

# Визуализация корреляционной матрицы с помощью heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={"shrink": .8})
plt.title('Корреляционная матрица признаков Iris')
plt.show()