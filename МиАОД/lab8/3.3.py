# Задание 3: Сравнение t-SNE и PCA.
# Используйте любой набор данных на ваше усмотрение. Примените PCA и t-SNE, чтобы снизить размерность до 2-х, и сравните разницы в результатах визуализации.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Загрузка данных
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# 2. Применение PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Применение t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 4. Визуализация результатов
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# PCA
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[0].scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.7, label=target_name)

ax[0].set_title('PCA of Wine Dataset')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
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