# Задание 5: Сравнение UMAP и t-SNE.
# Используйте один и тот же набор данных для применения UMAP и t-SNE. Сравните влияние этих методов снижения размерности на визуальное разделение классов в данных.
# Каждое задание должно включать в себя следующие шаги: загрузка и предварительная обработка данных, применение метода снижения размерности, и, при необходимости, обучение модели на полученных признаках и оценка производительности модели.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

# 1. Загрузка и предварительная обработка данных
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Применение UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# 3. Применение t-SNE
tsne_model = TSNE(n_components=2, random_state=42)
X_tsne = tsne_model.fit_transform(X_scaled)

# 4. Визуализация результатов
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# UMAP
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[0].scatter(X_umap[y == i, 0], X_umap[y == i, 1], color=color, alpha=0.7, label=target_name)

ax[0].set_title('UMAP of Wine Dataset')
ax[0].set_xlabel('UMAP Component 1')
ax[0].set_ylabel('UMAP Component 2')
ax[0].legend(loc='best')
ax[0].grid()

# t-SNE
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[1].scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], color=color, alpha=0.7, label=target_name)

ax[1].set_title('t-SNE of Wine Dataset')
ax[1].set_xlabel('t-SNE Component 1')
ax[1].set_ylabel('t-SNE Component 2')
ax[1].legend(loc='best')
ax[1].grid()

plt.tight_layout()
plt.show()