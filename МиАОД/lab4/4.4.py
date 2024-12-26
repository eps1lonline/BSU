import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
file_path = "C:/Users/nikit/Desktop/USD_Corrected_Month.txt"
data = pd.read_csv(file_path)

# Преобразование столбца 'Date' в формат даты
data['Date'] = pd.to_datetime(data['Date'])

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Value'], marker='o')
plt.title('Временной ряд USD')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# Показать график
plt.show()