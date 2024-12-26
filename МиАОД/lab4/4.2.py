import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv("C:/Users/nikit/Desktop/weather_description.csv")

# Преобразование столбца datetime в формат datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Пример: Вычисление частоты различных типов погоды
weather_counts = data.iloc[:, 1:].apply(pd.Series.value_counts).fillna(0)

# Создание тепловой карты корреляции
plt.figure(figsize=(12, 8))
sns.heatmap(weather_counts.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

# Настройка заголовка
plt.title('Тепловая карта корреляции между параметрами погоды')
plt.show()