import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из CSV файла
data = pd.read_csv("C:/Users/nikit/Desktop/car_date.csv")

# Обзор данных
print("Первые несколько строк данных:")
print(data.head())

# Проверка структуры данных
print("\nИнформация о данных:")
print(data.info())

# 1. Столбчатая диаграмма - Продаваемые цены по автомобилям
plt.figure(figsize=(10, 6))
sns.barplot(x='Car_Name', y='Selling_Price', data=data)
plt.title('Цена продажи автомобилей')
plt.xlabel('Название автомобиля')
plt.ylabel('Цена продажи (в лакхах)')
plt.xticks(rotation=45)
plt.show()

# 2. Гистограмма - Распределение пробега
plt.figure(figsize=(10, 6))
sns.histplot(data['Kms_Driven'], bins=10, kde=True)
plt.title('Распределение пробега')
plt.xlabel('Пробег (в км)')
plt.ylabel('Частота')
plt.show()

# 3. Линейный график - Продаваемая цена по годам
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Selling_Price', data=data, marker='o')
plt.title('Цена продажи по годам')
plt.xlabel('Год')
plt.ylabel('Цена продажи (в лакхах)')
plt.show()

# 4. Диаграмма размаха (ящик с усами) - Продаваемая цена
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=data)
plt.title('Диаграмма размаха цены продажи по типу топлива')
plt.xlabel('Тип топлива')
plt.ylabel('Цена продажи (в лакхах)')
plt.show()

# 5. Радиальная диаграмма (пироговая диаграмма) - Типы топлива
plt.figure(figsize=(8, 8))
fuel_counts = data['Fuel_Type'].value_counts()
plt.pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Распределение типов топлива')
plt.axis('equal')  # Равное соотношение сторон обеспечивает круговую форму диаграммы.
plt.show()

# 6. Диаграмма рассеяния - Продаваемая цена против пробега
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Kms_Driven', y='Selling_Price', data=data, hue='Fuel_Type', style='Transmission', s=100)
plt.title('Цена продажи против пробега')
plt.xlabel('Пробег (в км)')
plt.ylabel('Цена продажи (в лакхах)')
plt.show()