# Задание 2: Определение влияния параметров t-SNE.
# Используйте тот же набор данных "Iris". Примените t-SNE с разными значениями параметров (например, число итераций, learning rate) и сравните полученные результаты.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Функция для применения t-SNE и визуализации
def apply_tsne_and_plot(X, y, n_iter, learning_rate, title):
    tsne = TSNE(n_components=2, n_iter=n_iter, learning_rate=learning_rate, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    colors = ['navy', 'turquoise', 'darkorange']
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], color=color, alpha=0.7, label=target_name)

    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Применение t-SNE с различными параметрами

# 1. Стандартные параметры
apply_tsne_and_plot(X, y, n_iter=1000, learning_rate=200, title='t-SNE with n_iter=1000, learning_rate=200')

# 2. Увеличенное число итераций
apply_tsne_and_plot(X, y, n_iter=3000, learning_rate=200, title='t-SNE with n_iter=3000, learning_rate=200')

# 3. Уменьшенный learning rate
apply_tsne_and_plot(X, y, n_iter=1000, learning_rate=50, title='t-SNE with n_iter=1000, learning_rate=50')

# 4. Увеличенный learning rate
apply_tsne_and_plot(X, y, n_iter=1000, learning_rate=1000, title='t-SNE with n_iter=1000, learning_rate=1000')