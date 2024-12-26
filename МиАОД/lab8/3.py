# Задание 3: Исследование влияния предобработки данных на результаты PCA.
# Используйте набор данных "Boston Housing" из sklearn.datasets. 
# Примените различные методы предобработки (например, масштабирование, нормализацию) перед применением PCA и сравните полученные результаты.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# 1. Загрузка данных
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 2. Масштабирование
scaler_standard = StandardScaler()
X_standard_scaled = scaler_standard.fit_transform(X)

scaler_minmax = MinMaxScaler()
X_minmax_scaled = scaler_minmax.fit_transform(X)

# 3. Применение PCA к исходным и предобработанным данным
pca = PCA(n_components=2)

# PCA без предобработки
X_pca_original = pca.fit_transform(X)

# PCA с масштабированием
X_pca_standard = pca.fit_transform(X_standard_scaled)

# PCA с нормализацией
X_pca_minmax = pca.fit_transform(X_minmax_scaled)

# 4. Визуализация результатов
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Исходные данные
ax[0].scatter(X_pca_original[:, 0], X_pca_original[:, 1], alpha=0.7)
ax[0].set_title('PCA без предобработки')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
ax[0].grid()

# Масштабирование
ax[1].scatter(X_pca_standard[:, 0], X_pca_standard[:, 1], alpha=0.7, color='orange')
ax[1].set_title('PCA с масштабированием (StandardScaler)')
ax[1].set_xlabel('Principal Component 1')
ax[1].set_ylabel('Principal Component 2')
ax[1].grid()

# Нормализация
ax[2].scatter(X_pca_minmax[:, 0], X_pca_minmax[:, 1], alpha=0.7, color='green')
ax[2].set_title('PCA с нормализацией (MinMaxScaler)')
ax[2].set_xlabel('Principal Component 1')
ax[2].set_ylabel('Principal Component 2')
ax[2].grid()

plt.tight_layout()
plt.show()