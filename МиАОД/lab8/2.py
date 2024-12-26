# Задание 2: Сравнение PCA и Factor Analysis.
# Используйте набор данных "Wine" из sklearn.datasets. Примените PCA и Factor Analysis, чтобы снизить размерность до 2-х и визуализируйте различия в результатах. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, FactorAnalysis

# 1. Загрузка данных
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# 2. Применение PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Применение Factor Analysis
fa = FactorAnalysis(n_components=2)
X_fa = fa.fit_transform(X)

# 4. Визуализация результатов
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# PCA
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[0].scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, label=target_name)
ax[0].set_title('PCA of Wine dataset')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
ax[0].legend(loc='best')
ax[0].grid()

# Factor Analysis
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[1].scatter(X_fa[y == i, 0], X_fa[y == i, 1], color=color, alpha=.8, label=target_name)
ax[1].set_title('Factor Analysis of Wine dataset')
ax[1].set_xlabel('Factor 1')
ax[1].set_ylabel('Factor 2')
ax[1].legend(loc='best')
ax[1].grid()

plt.tight_layout()
plt.show()