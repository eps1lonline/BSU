# Задание 1: Рекурсивное исключение признаков (RFE).
# Используйте набор данных "Iris" из sklearn.datasets. 
# Примените метод RFE с использованием модели логистической регрессии. 
# Укажите количество признаков для выбора и сравните производительность модели с и без этих признаков.

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

# 1. Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# 2. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Создание модели логистической регрессии
model = LogisticRegression(max_iter=200)

# 4. Применение RFE для выбора 2 признаков
rfe = RFE(model, n_features_to_select=2)
rfe.fit(X_train, y_train)

# Получение выбранных признаков
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# 5. Обучение модели на выбранных признаках
model.fit(X_train_rfe, y_train)
y_pred_rfe = model.predict(X_test_rfe)

# Оценка производительности модели с выбранными признаками
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)

# 6. Обучение модели на всех признаках
model.fit(X_train, y_train)
y_pred_full = model.predict(X_test)

# Оценка производительности модели без исключения признаков
accuracy_full = accuracy_score(y_test, y_pred_full)

# Вывод результатов
print(f'Accuracy with RFE: {accuracy_rfe:.2f}')
print(f'Accuracy without RFE: {accuracy_full:.2f}')