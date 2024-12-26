# Задание 4: Применение Disciminant Analysis.
# Используйте набор данных "Iris" из sklearn.datasets. 
# Примените Linear Discriminant Analysis (LDA) и Quadratic Discriminant Analysis (QDA), чтобы снизить размерность до 2-х и визуализируйте результаты.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# 1. Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Применение LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# 3. Применение QDA
qda = QuadraticDiscriminantAnalysis()
X_qda = qda.fit(X, y).predict(X)

# 4. Визуализация результатов
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# LDA
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[0].scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color, alpha=.8, label=target_name)
ax[0].set_title('LDA of Iris dataset')
ax[0].set_xlabel('Linear Discriminant 1')
ax[0].set_ylabel('Linear Discriminant 2')
ax[0].legend(loc='best')
ax[0].grid()

# QDA (поскольку QDA возвращает классы, а не компоненты, нам нужно создать новый набор данных для визуализации)
X_qda_2d = lda.transform(X)  # Применяем LDA к исходным данным для 2D визуализации
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax[1].scatter(X_qda_2d[y == i, 0], X_qda_2d[y == i, 1], color=color, alpha=.8, label=target_name)
ax[1].set_title('QDA of Iris dataset (using LDA for visualization)')
ax[1].set_xlabel('Quadratic Discriminant 1')
ax[1].set_ylabel('Quadratic Discriminant 2')
ax[1].legend(loc='best')
ax[1].grid()

plt.tight_layout()
plt.show()