# Импортируем необходимые библиотеки
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Генерация двух шкалированных наборов данных
np.random.seed(42)
data1 = np.random.normal(loc=75, scale=10, size=100)  # Среднее 75, стандартное отклонение 10
data2 = np.random.normal(loc=80, scale=12, size=100)  # Среднее 80, стандартное отклонение 12

# Проверка гипотезы с использованием t-теста
t_stat, p_value = stats.ttest_ind(data1, data2)

# Визуализация данных
plt.figure(figsize=(8, 6))
sns.histplot(data1, color='blue', label='Data 1 (mean=75)', kde=True, bins=15)
sns.histplot(data2, color='red', label='Data 2 (mean=80)', kde=True, bins=15)
plt.title('Distribution of Two Data Sets')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Выводим результаты t-теста
{
    'T-Statistic': t_stat,
    'P-Value': p_value
}