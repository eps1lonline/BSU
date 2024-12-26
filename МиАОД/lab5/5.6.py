import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.read_csv("C:/Users/nikit/Desktop/winequality-red.csv")

# Выделение признаков (feature matrix)
X = data.drop('quality', axis=1)

# Стандартизация
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)

# Нормализация (MinMaxScaler)
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

# Нормализация (RobustScaler)
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)

# Преобразование обратно в df для сравнения
X_standard_df = pd.DataFrame(X_standard, columns=X.columns)
X_minmax_df = pd.DataFrame(X_minmax, columns=X.columns)
X_robust_df = pd.DataFrame(X_robust, columns=X.columns)

print("Оригинальные данные:")
print(X.head(5))

print("\nСтандартизированные данные (StandardScaler):")
print(X_standard_df.head(5))

print("\nНормализованные данные (MinMaxScaler):")
print(X_minmax_df.head(5))

print("\nНормализованные данные (RobustScaler):")
print(X_robust_df.head(5))
