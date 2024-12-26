# Задание 5: Сравнение PCA и LDA.
# Используйте любой набор данных с классифицирующей моделью. 
# Примените PCA и LDA и сравните, как влияют эти методы снижения размерности на эффективность классификации.from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Загрузка данных
wine = load_wine()
X = wine.data
y = wine.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Применение PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 3. Обучение классификатора на данных с PCA
classifier_pca = SVC(kernel='linear')
classifier_pca.fit(X_train_pca, y_train)
y_pred_pca = classifier_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# 4. Применение LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# 5. Обучение классификатора на данных с LDA
classifier_lda = SVC(kernel='linear')
classifier_lda.fit(X_train_lda, y_train)
y_pred_lda = classifier_lda.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)

# 6. Вывод результатов
print(f'Accuracy with PCA: {accuracy_pca:.2f}')
print(f'Accuracy with LDA: {accuracy_lda:.2f}')

# 7. Визуализация результатов
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# PCA
ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
ax[0].set_title('PCA - Wine Dataset')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')

# LDA
ax[1].scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
ax[1].set_title('LDA - Wine Dataset')
ax[1].set_xlabel('Linear Discriminant 1')
ax[1].set_ylabel('Linear Discriminant 2')

plt.tight_layout()
plt.show()