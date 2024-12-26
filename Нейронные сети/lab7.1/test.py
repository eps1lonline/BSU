import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
from keras.datasets import cifar10
from keras.utils import to_categorical

# Загрузка модели из JSON-файла
with open("cifar10_model.json", "r") as json_file:
    model_json = json_file.read()
    
# Десериализация JSON в модель
loaded_model = model_from_json(model_json)

# Загружаем веса в модель
loaded_model.load_weights("cifar10_model.weights.h5")

# Компилируем модель
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print("Модель успешно загружена и скомпилирована.")

# Загружаем данные CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Нормализация данных
X_test = X_test.astype('float32') / 255

# Преобразуем метки в категории
Y_test = to_categorical(y_test, 10)

# Оценка модели
scores = loaded_model.evaluate(X_test, Y_test, verbose=2)

# Выводим точность модели
print("Точность модели на тестовых данных: %.2f%%" % (scores[1] * 100))

# Обработка изображения для предсказания
im = Image.open('44.png')
im = im.resize((32, 32))  # Изменяем размер на 32x32
im_array = np.array(im)

# Убираем дополнительный канал, если присутствует
if im_array.shape[2] == 4:
    im_array = im_array[:, :, :3]  # Берем только первые 3 канала

# Нормализуем изображение
im_array = im_array.astype('float32') / 255.0

# Изменяем форму для предсказания
x = np.expand_dims(im_array, axis=0)  # Добавляем batch-размерность (1, 32, 32, 3)

# Нейронная сеть предсказывает класс изображения
prediction = loaded_model.predict(x)
print(prediction)

# Преобразуем ответ из категориального представления в метку класса
predicted_class = np.argmax(prediction, axis=1)

# Печатаем результат
print("Предсказанный класс:", predicted_class[0])