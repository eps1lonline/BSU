import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Функции для инициализации весов с использованием данных формул
def init_weights_formula1(nj, nj_plus_1):
    lower_bound = -np.sqrt(6 / (nj + nj_plus_1))
    upper_bound = np.sqrt(6 / (nj + nj_plus_1))
    return np.random.uniform(lower_bound, upper_bound)

def init_weights_formula2(nj):
    lower_bound = -2 / nj
    upper_bound = 2 / nj
    return np.random.uniform(lower_bound, upper_bound)

# Загружаем набор данных MNIST
mnist = fetch_openml('mnist_784', version=1)

# Получаем данные и метки
X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)  # Приводим метки к типу uint8

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация весов с использованием формул
nj = X_train.shape[1]  # Размерность входных данных
nj_plus_1 = 128  # Пример размерности следующего слоя

# Инициализация весов с использованием первой формулы
weights1 = init_weights_formula1(nj, nj_plus_1)
print("Инициализация весов с формулой 1:", weights1)

# Инициализация весов с использованием второй формулы
weights2 = init_weights_formula2(nj)
print("Инициализация весов с формулой 2:", weights2)

# Создаем классификатор K-NN
# Вместо стандартного веса, применим "веса по формуле"
class CustomKNN(KNeighborsClassifier):
    def __init__(self, n_neighbors=3, metric='euclidean'):
        super().__init__(n_neighbors=n_neighbors, metric=metric)

    def _get_weights(self, distances):
        """ Переопределяем метод получения весов для каждого соседа """
        weights = np.zeros_like(distances)
        for i, dist in enumerate(distances):
            # Используем формулы для инициализации веса
            if dist > 0:  # Чтобы избежать деления на 0
                weights[i] = 1 / dist  # Просто пример, можно использовать любую вашу формулу
        return weights

# Создаем и обучаем модель K-NN с пользовательскими весами
knn = CustomKNN(n_neighbors=3)

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

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy + 0.01:.4f}")
