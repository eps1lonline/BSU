import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)

# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование размерности изображений
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Преобразуем метки в категории
Y_train = to_categorical(y_train, 10)  # Обновлено
Y_test = to_categorical(y_test, 10)    # Обновлено

# Создаем последовательную модель
model = Sequential()

# Добавляем уровни сети
model.add(Dense(1200, input_dim=784, activation="relu", kernel_initializer="normal"))
model.add(Dense(1200, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))

# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

# Обучаем сеть
model.fit(X_train, Y_train, batch_size=50, epochs=2, validation_split=0.2, verbose=2)

# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))