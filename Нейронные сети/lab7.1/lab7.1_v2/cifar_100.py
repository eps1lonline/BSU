import numpy as np
import time
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD

# Задаем seed для повторяемости результатов
np.random.seed(42)

# Загружаем данные
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
# Нормализуем данные
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
Y_train = to_categorical(y_train, 100)  # Увеличиваем количество классов до 100
Y_test = to_categorical(y_test, 100)

# Параметры
batch_size = 32
nb_classes = 100  # Увеличиваем количество классов до 100
nb_epoch = 10

# Создаем модель
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))

# Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# Четвертый сверточный слой
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
model.add(Dense(nb_classes, activation='softmax'))

# Задаем параметры оптимизации
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Измеряем время обучения
start_time = time.time()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1, shuffle=True, verbose=2)
training_time = time.time() - start_time

# Оцениваем качество обучения модели на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
accuracy = scores[1] * 100

# Выводим результаты
print(f"Время обучения: {training_time:.2f} сек, Точность: {accuracy:.2f}%")

# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("cifar100_model.json", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("cifar100_model.weights.h5")