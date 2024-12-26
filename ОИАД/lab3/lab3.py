import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

data = pd.read_csv('airquality.csv')

print("Количество пропусков в данных:\n", data.isnull().sum())
data = data.dropna()

# 2. Построить линейную регрессионную модель зависимости уровня озона от других факторов
X = data[['Solar.R', 'Wind', 'Temp']]
y = data['Ozone']
X = sm.add_constant(X)

try:
    linear_model = sm.OLS(y, X).fit()
    print(linear_model.summary())
except Exception as e:
    print("Ошибка при создании модели:", e)

# 3. Оценить значимость коэффициентов модели
X_reduced = data[['Solar.R', 'Temp']]
X_reduced = sm.add_constant(X_reduced)

try:
    linear_model_reduced = sm.OLS(y, X_reduced).fit()
    print(linear_model_reduced.summary())
except Exception as e:
    print("Ошибка при создании уменьшенной модели:", e)

# 4. Построить нелинейную регрессионную модель
poly = PolynomialFeatures(degree=2) 
X_poly = poly.fit_transform(data[['Solar.R', 'Wind', 'Temp']])
poly_model = LinearRegression().fit(X_poly, y)

# 5. Провести анализ остатков
residuals_linear = linear_model_reduced.resid
residuals_poly = y - poly_model.predict(X_poly)

# Установим стиль
sns.set(style="whitegrid")

plt.figure(figsize=(14, 6))

# Гистограмма остатков линейной модели
plt.subplot(1, 2, 1)
sns.histplot(residuals_linear, kde=True, color='skyblue', bins=30, edgecolor='black')
plt.title('Остатки линейной модели', fontsize=16)
plt.xlabel('Значения остатков', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--')  # Вертикальная линия на нуле
plt.grid(True)

# Гистограмма остатков полиномиальной модели
plt.subplot(1, 2, 2)
sns.histplot(residuals_poly, kde=True, color='salmon', bins=30, edgecolor='black')
plt.title('Остатки полиномиальной модели', fontsize=16)
plt.xlabel('Значения остатков', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--')  # Вертикальная линия на нуле
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Получить прогнозные значения по каждой модели
predictions_linear = linear_model_reduced.predict(X_reduced)
predictions_poly = poly_model.predict(X_poly)

# 7. Представить результаты графически
plt.figure(figsize=(12, 6))

# Линейная модель
plt.subplot(1, 2, 1)
plt.scatter(data['Solar.R'], y, color='green', label='Фактические данные')
plt.scatter(data['Solar.R'], predictions_linear, color='red', label='Прогноз')
plt.title('Линейная модель')
plt.xlabel('Solar.R')
plt.ylabel('Ozone')
plt.legend()

# Полиномиальная модель
plt.subplot(1, 2, 2)
plt.scatter(data['Solar.R'], y, color='green', label='Фактические данные')
plt.scatter(data['Solar.R'], predictions_poly, color='red', label='Прогноз')
plt.title('Полиномиальная модель')
plt.xlabel('Solar.R')
plt.ylabel('Ozone')
plt.legend()

plt.tight_layout()
plt.show()