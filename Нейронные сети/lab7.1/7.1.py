import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Устанавливаем seed для воспроизводимости
np.random.seed(42)

# Загружаем данные
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Нормализуем данные
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Преобразуем метки в категории
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

# Создаём последовательную модель
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu'))
# Слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# Четвёртый сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.5))
# Выходной полносвязный слой
model.add(Dense(10, activation='softmax'))

# Компилируем модель
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# Обучаем модель
model.fit(X_train, Y_train, batch_size=200, epochs=10, validation_split=0.2, shuffle=True, verbose=2)

# Оцениваем качество модели на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)

# Печатаем точность
print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))

# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("cifar10_model.json", "w")
json_file.write(model_json)
json_file.close()
model.save_weights("cifar10_model.weights.h5")

print ("Сохранили Model")

