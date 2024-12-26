# Импорт необходимых библиотек
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, mean_squared_error

# Загрузка набора данных о цветах ирисов
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Отбор признаков с помощью взаимной информации
mi = mutual_info_classif(X_train, y_train)
mi_indices = mi.argsort()[::-1][:2]  # Выбираем 2 наиболее информативных признака
X_train_mi = X_train.iloc[:, mi_indices]
X_test_mi = X_test.iloc[:, mi_indices]

# Обучение модели с выбранными признаками
model_mi = LogisticRegression(max_iter=200)
model_mi.fit(X_train_mi, y_train)
y_pred_mi = model_mi.predict(X_test_mi)
accuracy_mi = accuracy_score(y_test, y_pred_mi)

# 2. Отбор признаков с помощью RFE
model_rfe = LogisticRegression(max_iter=200)
rfe = RFE(estimator=model_rfe, n_features_to_select=2)
rfe.fit(X_train, y_train)
X_train_rfe = X_train.iloc[:, rfe.support_]
X_test_rfe = X_test.iloc[:, rfe.support_]

# Обучение модели с выбранными признаками
model_rfe.fit(X_train_rfe, y_train)
y_pred_rfe = model_rfe.predict(X_test_rfe)
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)

# 3. Отбор признаков с помощью Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_coefficients = lasso.coef_

# Выбираем признаки, у которых коэффициенты не равны нулю
selected_features_lasso = [i for i, coef in enumerate(lasso_coefficients) if coef != 0]
X_train_lasso = X_train.iloc[:, selected_features_lasso]
X_test_lasso = X_test.iloc[:, selected_features_lasso]

# Обучение модели с выбранными признаками
model_lasso = LogisticRegression(max_iter=200)
model_lasso.fit(X_train_lasso, y_train)
y_pred_lasso = model_lasso.predict(X_test_lasso)
accuracy_lasso = accuracy_score(y_test, y_pred_lasso)

# Вывод результатов
print("Точность моделей с различными методами отбора признаков:")
print(f"1. Взаимная информация: {accuracy_mi:.2f}")
print(f"2. Рекурсивное исключение признаков (RFE): {accuracy_rfe:.2f}")
print(f"3. Lasso регрессия: {accuracy_lasso:.2f}")