# Регрессия с использованием метода эластичной сети.
# На том же наборе данных обучите ElasticNet, который объединяет L1 и L2 регуляризацию. Экспериментируйте с разными соотношениями L1 и L2 регуляризации и
# установите, как это влияет на производительность модели.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Шаг 1: Загрузка набора данных
data = load_diabetes()
X = data.data
y = data.target

# Добавим дополнительные синтетические признаки
np.random.seed(42)
X = np.hstack((X, np.random.randn(X.shape[0], 10)))  # Добавляем 10 случайных признаков

# Шаг 2: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Определение диапазонов alpha и l1_ratio
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
l1_ratios = [0.1, 0.5, 0.9]  # 0.1 - больше L2, 0.9 - больше L1

# Шаг 4: Обучение моделей ElasticNet и оценка производительности
results = []

for alpha in alphas:
    for l1_ratio in l1_ratios:
        elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        elastic_net_model.fit(X_train, y_train)
        predictions = elastic_net_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        results.append({
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'MSE': mse,
            'Coefficients': elastic_net_model.coef_
        })

# Преобразование результатов в DataFrame
results_df = pd.DataFrame(results)

# Шаг 5: Визуализация производительности
plt.figure(figsize=(12, 6))
for l1_ratio in l1_ratios:
    mse_values = results_df[results_df['l1_ratio'] == l1_ratio]['MSE']
    plt.plot(results_df[results_df['l1_ratio'] == l1_ratio]['alpha'], mse_values, marker='o', label=f'l1_ratio = {l1_ratio}')

plt.xscale('log')
plt.xlabel('Regularization Strength (alpha)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of L1 Ratio on ElasticNet Performance')
plt.legend()
plt.grid()
plt.show()

# Шаг 6: Анализ распределения весов признаков
coeffs = np.array([result['Coefficients'] for result in results])

plt.figure(figsize=(14, 7))
for i in range(coeffs.shape[1]):
    for j, l1_ratio in enumerate(l1_ratios):
        plt.plot(results_df[results_df['l1_ratio'] == l1_ratio]['alpha'], coeffs[:, i][results_df['l1_ratio'] == l1_ratio], label=f'Feature {i} - l1_ratio {l1_ratio}', linestyle='--' if j > 0 else '-')
        
plt.xscale('log')
plt.xlabel('Regularization Strength (alpha)')
plt.ylabel('Coefficient Value')
plt.title('Effect of Regularization on ElasticNet Coefficients')
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.grid()
plt.show()