# это таблица, показывающая коэффициенты корреляции между множеством переменных. Она используется для анализа взаимосвязей между переменными в наборе данных.
# # взаимосвязей между переменными в наборе данных
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Загрузка данных о винах
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# Рассчет корреляционной матрицы
corr_matrix = X.corr()

# Установка размера графика
plt.figure(figsize=(10, 8))

# Построение тепловой карты
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Заголовок
plt.title('Корреляционная матрица для датасета о винах')

# Показать график
plt.show()

