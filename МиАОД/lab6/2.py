# Важность признака в моделях, определяется на основе того, как сильно каждый признак влияет на качество предсказаний модели
# # Импорт необходимых библиотек
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузка данных о винах
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# Шаг 2: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Подгонка модели RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Шаг 4: Получение важности признаков
importances = model.feature_importances_

# Создание DataFrame для визуализации важности признаков
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Сортировка по важности
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Шаг 5: Визуализация важности признаков
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Важность признаков в Random Forest')
plt.xlabel('Важность')
plt.ylabel('Признаки')
plt.show()

# Вывод важности признаков
print(feature_importances)