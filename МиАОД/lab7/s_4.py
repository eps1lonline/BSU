# Задание 4: Сравнение методов-оберток.
# Используйте один и тот же набор данных для применения различных методов-оберток для отбора признаков, 
# например, RFE и Sequential Feature Selector, и сравните полученные результаты.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score

# Загружаем данные
iris = load_iris()
X, y = iris.data, iris.target

# Разделяем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Инициализируем классификатор
svc = SVC(kernel="linear")

# Метод RFE
rfe = RFE(estimator=svc, n_features_to_select=2)
rfe.fit(X_train, y_train)
rfe_selected_features = rfe.support_

# Применяем отобранные признаки
X_train_rfe = X_train[:, rfe_selected_features]
X_test_rfe = X_test[:, rfe_selected_features]

# Обучаем и оцениваем модель
svc.fit(X_train_rfe, y_train)
rfe_accuracy = accuracy_score(y_test, svc.predict(X_test_rfe))

# Sequential Feature Selector
sfs = SequentialFeatureSelector(svc, n_features_to_select=2, direction='forward')
sfs.fit(X_train, y_train)
sfs_selected_features = sfs.get_support()

# Применяем отобранные признаки
X_train_sfs = X_train[:, sfs_selected_features]
X_test_sfs = X_test[:, sfs_selected_features]

# Обучаем и оцениваем модель
svc.fit(X_train_sfs, y_train)
sfs_accuracy = accuracy_score(y_test, svc.predict(X_test_sfs))

# Результаты
print("RFE выбранные признаки:", rfe_selected_features)
print("Точность с RFE:", rfe_accuracy)

print("SFS выбранные признаки:", sfs_selected_features)
print("Точность с SFS:", sfs_accuracy)