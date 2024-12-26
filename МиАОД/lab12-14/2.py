import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Чтение данных из CSV-файла
data = pd.read_csv('creditcard.csv')  # Укажите путь к вашему файлу

# Просмотр первых строк данных
print(data.head())

# Проверка на пропущенные значения
print(data.isnull().sum())

# Разделение данных на признаки и целевую переменную
X = data.drop(['Time', 'Class'], axis=1)  # Удаляем ненужные признаки
y = data['Class'].astype(int)  # Приводим целевую переменную к типу int

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Создание и обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка модели
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанная метка')
plt.ylabel('Истинная метка')
plt.title('Матрица ошибок')
plt.show()