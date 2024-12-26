# Задание 2: Sequential Feature Selector.
# Используйте набор данных "Boston Housing" из sklearn.datasets. 
# Используйте Sequential Feature Selector для выбора признаков с использованием модели Random Forest. 
# Визуализируйте "важность" признаков.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузка данных
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Шаг 2: Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Инициализация модели Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Шаг 4: Инициализация и обучение Sequential Feature Selector
sfs = SequentialFeatureSelector(model, n_features_to_select='auto', direction='forward')
sfs.fit(X_train, y_train)

# Получение выбранных признаков
selected_features = X.columns[sfs.get_support()]
print("Выбранные признаки:", selected_features.tolist())

# Шаг 5: Обучение модели на всех признаках
model.fit(X_train, y_train)

# Получение важности признаков
importances = model.feature_importances_

# Шаг 6: Создание DataFrame для визуализации
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Шаг 7: Визуализация
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Важность признаков')
plt.xlabel('Важность')
plt.ylabel('Признаки')
plt.show()