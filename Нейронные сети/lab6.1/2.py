import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from PIL import Image

# Загрузка модели из JSON-файла
with open("mnist_model.json", "r") as json_file:
    model_json = json_file.read()
    
# Десериализация JSON в модель
loaded_model = model_from_json(model_json)

# Загружаем веса в модель
loaded_model.load_weights("mnist_model.weights.h5")

# Компилируем модель
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print("Модель успешно загружена и скомпилирована.")

# Теперь можно использовать модель для оценки или предсказаний
np.random.seed(42)

# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование размерности изображений
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Нормализация данных
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Преобразуем метки в категории
Y_train = to_categorical(y_train, 10)  
Y_test = to_categorical(y_test, 10)    

scores = loaded_model.evaluate(X_test, Y_test, verbose=2)
# Выводим точность модели
print("Точность модели на тестовых данных: %.2f%%" % (scores[1] * 100))

# Обработка изображения для предсказания
im = Image.open('2.png')
im_grey = im.convert('L')
im_array = np.array(im_grey)

# Изменение размера массива изображения
im_array = np.reshape(im_array, (1, 784)).astype('float32')

# Инвертируем изображение
x = 255 - im_array

# Нормализуем изображение
x /= 255

# Нейронная сеть предсказывает класс изображения
prediction = loaded_model.predict(x)  # Исправлено здесь
print(prediction)

# Преобразуем ответ из категориального представления в метку класса
predicted_class = np.argmax(prediction, axis=1)

# Печатаем результат
print(predicted_class)