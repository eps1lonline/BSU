import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro
from scipy.optimize import curve_fit
import pandas as pd
# кэф. детерминации

# 1. Генерация данных
np.random.seed(20)
num_points = 1000
x = np.linspace(1, 5, num_points)  #[1, 20]
y_true = np.exp(x) + np.cos(x) + np.sin(x)
noise = np.random.normal(0, 10, num_points)
y_noisy = y_true + noise 

plt.figure(figsize=(10, 6))
plt.scatter(x, y_noisy, label="Шумные данные", color="blue", s=10)
plt.plot(x, y_true, label="Истинная зависимость", color="red", linewidth=2)
plt.title("Исходные данные")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# 2. Построение линейной регрессионной модели: yi = b0 + b1 * xi + εi
x_reshaped = x.reshape(-1, 1)
linear_model = LinearRegression()
linear_model.fit(x_reshaped, y_noisy)
y_pred_linear = linear_model.predict(x_reshaped)

# 3. Линейная регрессия с преобразованием f(x): yi = b0 + b1 * f(xi)
def f(x):
    return np.log(x + 1)

x_transformed = f(x).reshape(-1, 1)
linear_model_f = LinearRegression()
linear_model_f.fit(x_transformed, y_noisy)
y_pred_f = linear_model_f.predict(x_transformed)

# 4. Построение нелинейной модели, например, yi = b0 + b1 * xi^b2

def nonlinear_model(x, b0, b1, b2):
    return b0 + b1 * x**b2

# Оценка параметров нелинейной модели
try:
    params, _ = curve_fit(nonlinear_model, x, y_noisy, p0=[1, 1, 0.5], maxfev=10000)
    y_pred_nonlinear = nonlinear_model(x, *params)
except RuntimeError as e:
    print(f"Ошибка при оптимизации: {e}")
    params = [np.nan, np.nan, np.nan]
    y_pred_nonlinear = np.full_like(x, np.nan)


# 5. Анализ остатков на нормальность
def analyze_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    stat, p = shapiro(residuals)  # Тест Шапиро-Уилка
    print(f"Остатки для {model_name}:")
    print(f"  Среднеквадратичная ошибка (MSE): {mean_squared_error(y_true, y_pred):.4f}")
    print(f"  Коэффициент детерминации (R^2): {r2_score(y_true, y_pred):.4f}")
    print(f"  Нормальность остатков (p-value): {p:.4f}\n")
    return mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)

print("\nАнализ остатков:")
mse_linear, r2_linear = analyze_residuals(y_noisy, y_pred_linear, "Линейная модель")
mse_f, r2_f = analyze_residuals(y_noisy, y_pred_f, "Линейная модель с f(x)")
mse_nonlinear, r2_nonlinear = analyze_residuals(y_noisy, y_pred_nonlinear, "Нелинейная модель")

# 6. Сводная таблица
summary = pd.DataFrame({
    "Модель": ["Линейная", "Линейная с f(x)", "Нелинейная"],
    "MSE": [mse_linear, mse_f, mse_nonlinear],
    "R^2": [r2_linear, r2_f, r2_nonlinear]
})
print("\nСводная таблица:")
print(summary)

# 7. Прогноз для точки b + 1 (x = 6)
x_new = np.array([[6]])
y_new_linear = linear_model.predict(x_new)
y_new_f = linear_model_f.predict(f(x_new).reshape(-1, 1))
y_new_nonlinear = nonlinear_model(6, *params)

print("\nПрогнозы для точки x = 6:")
print(f"  Линейная модель: {y_new_linear[0]:.4f}")
print(f"  Линейная модель с f(x): {y_new_f[0]:.4f}")
print(f"  Нелинейная модель: {y_new_nonlinear:.4f}")

# 8. Графики всех моделей и прогноза
print(f"Прогноз")
plt.figure(figsize=(12, 8))
plt.scatter(x, y_noisy, label="Шумные данные", color="blue", s=10)
plt.plot(x, y_pred_linear, label="Линейная модель", color="green", linewidth=2)
plt.plot(x, y_pred_f, label="Линейная модель с f(x)", color="orange", linewidth=2)
plt.plot(x, y_pred_nonlinear, label="Нелинейная модель", color="purple", linewidth=2)
plt.scatter(21, y_new_linear, color="green", label=f"Прогноз (Линейная): {y_new_linear[0]:.2f}", s=100, marker="x")
plt.scatter(21, y_new_f, color="orange", label=f"Прогноз (f(x)): {y_new_f[0]:.2f}", s=100, marker="x")
plt.scatter(21, y_new_nonlinear, color="purple", label=f"Прогноз (Нелинейная): {y_new_nonlinear:.2f}", s=100, marker="x")
plt.title("Сравнение моделей и прогнозов")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()