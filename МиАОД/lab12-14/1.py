import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Загружаем набор данных MNIST
mnist = fetch_openml('mnist_784', version=1)

# Получаем данные и метки
X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)  # Приводим метки к типу uint8

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем классификатор K-NN
knn = KNeighborsClassifier(n_neighbors=3)

# Обучаем модель
knn.fit(X_train, y_train)

# Делаем предсказания на тестовой выборке
y_pred = knn.predict(X_test)

# Оцениваем модель
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# Визуализация некоторых предсказаний
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {y_pred[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")
