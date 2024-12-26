import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/nikit/Desktop/train.csv")

numeric_data = data.select_dtypes(include=[float, int])

descriptive_stats = numeric_data.describe()
print("Основные метрики описательной статистики:")
print(descriptive_stats)

correlation_matrix = numeric_data.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Тепловая карта корреляций между переменными')
plt.show()