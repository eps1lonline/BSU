import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/nikit/Desktop/diabetes.csv")

missing_values = data.isnull().sum()
print("Пропущенные значения в каждом столбце:\n", missing_values)

# Заполнение пропущенных значений медианой
data_filled_median = data.fillna(data.median())
print("\nПропущенные значения после заполнения медианой:\n", data_filled_median.isnull().sum())

# Разделение данных на признаки и целевую переменную
X = data_filled_median.drop("Outcome", axis=1)
y = data_filled_median["Outcome"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование пропущенных значений
predicted_values = model.predict(X_test)
print("\nПрогнозируемые значения:\n", predicted_values)

# Сравнение результатов
print("\nОригинальный размер данных:", data.shape)
print("Размер после заполнения медианой:", data_filled_median.shape)