# Задание 4: Применение Ранговой корреляции Спирмена.
# Используйте любой набор данных, имеющий порядковые признаки. 
# Примените корреляцию Спирмена для выбора наиболее значимых признаков.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Шаг 1: Загрузка данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Шаг 2: Применение корреляции Спирмена
spearman_corr = data.corr(method='spearman')

# Шаг 3: Визуализация корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Корреляция Спирмена между признаками')
plt.show()

# Шаг 4: Анализ значимых признаков
# Например, выводим корреляцию с целевой переменной 'quality'
target_corr = spearman_corr['quality'].sort_values(ascending=False)
print("Корреляция признаков с качеством вина:")
print(target_corr)