import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Шаг 1: Загрузка данных
file_path = "C:/Users/nikit/Desktop/Summary of Weather.csv"
data = pd.read_csv(file_path)

# Удаление строк с пропусками в MeanTemp
data = data.dropna(subset=['MeanTemp'])

# Шаг 2: Предварительный анализ данных
print(data.head())
print(data.describe())

# Шаг 3: Обнаружение аномалий
data['Z-Score'] = stats.zscore(data['MeanTemp'])

# Найдите аномалии с Z-оценкой > 2 или < -2
anomalies = data[(data['Z-Score'] > 2) | (data['Z-Score'] < -2)]
print("Аномальные значения:")
print(anomalies)

# Шаг 4: Визуализация
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['MeanTemp'], label='Средняя температура', color='blue')
plt.scatter(anomalies['Date'], anomalies['MeanTemp'], color='red', label='Аномалии')
plt.title('Аномалии температуры с течением времени')
plt.xlabel('Дата')
plt.ylabel('Средняя температура')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()