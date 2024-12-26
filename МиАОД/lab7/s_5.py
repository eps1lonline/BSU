# Задание 5: Анализ предсказательной способности признаков.
# Используйте набор данных "Wine" из sklearn.datasets. 
# Выберите модель машинного обучения и метод-обертку для отбора признаков и исследуйте, как влияет отбор признаков на предсказательную способность модели

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

# Загружаем данные
wine = load_wine()
X, y = wine.data, wine.target

# Разделяем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Инициализируем классификатор
rf = RandomForestClassifier(random_state=42)

# Обучение модели без отбора признаков
rf.fit(X_train, y_train)
initial_accuracy = accuracy_score(y_test, rf.predict(X_test))

# Метод RFE
rfe = RFE(estimator=rf, n_features_to_select=5)
rfe.fit(X_train, y_train)
rfe_selected_features = rfe.support_

# Применяем отобранные признаки
X_train_rfe = X_train[:, rfe_selected_features]
X_test_rfe = X_test[:, rfe_selected_features]

# Обучение модели с отобранными признаками
rf.fit(X_train_rfe, y_train)
rfe_accuracy = accuracy_score(y_test, rf.predict(X_test_rfe))

# Результаты
print("Точность без отбора признаков:", initial_accuracy)
print("Точность с RFE:", rfe_accuracy)
print("RFE выбранные признаки:", rfe_selected_features)