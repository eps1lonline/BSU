# это метод отбора признаков, который последовательно удаляет наименее значимые признаки, чтобы улучшить производительность модели
# # Импорт необходимых библиотек
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# Загрузка набора данных о цветах ирисов
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели линейной регрессии
model = LinearRegression()

# Создание объекта RFE для выбора признаков
rfe = RFE(estimator=model, n_features_to_select=2)  # Выбираем 2 признака
rfe.fit(X_train, y_train)

# Вывод результатов
selected_features = X.columns[rfe.support_]
print("Выбранные признаки с помощью RFE:")
print(selected_features)

# Оценка модели на тестовых данных
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
model.fit(X_train_selected, y_train)

# Оценка точности модели
score = model.score(X_test_selected, y_test)
print(f"\nТочность модели с выбранными признаками: {score:.2f}")