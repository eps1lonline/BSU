import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.datasets import imdb

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)

# Максимальное количество слов (по частоте использования)
max_features = 5000
# Максимальная длина рецензии в словах
maxlen = 80

# Загружаем данные
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Заполняем или обрезаем рецензии
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Создаем сеть
model = Sequential()
# Слой для векторного представления слов
model.add(Embedding(max_features, 32))
model.add(SpatialDropout1D(0.2))
# Слой долго-краткосрочной памяти
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# Полносвязный слой
model.add(Dense(1, activation='sigmoid'))

# Компилируем модель
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Обучаем модель
model.fit(X_train, y_train, batch_size=64, epochs=1,
          validation_data=(X_test, y_test), verbose=2)

# Проверяем качество обучения на тестовых данных
scores = model.evaluate(X_test, y_test, batch_size=64)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))

# Генерируем описание модели в формате JSON
model_json = model.to_json()
# Записываем модель в файл
with open("cc_model.json", "w") as json_file:
    json_file.write(model_json)
# Сохраняем веса модели с правильным расширением
model.save_weights("cc_model.weights.h5")

print("Сохранили Model")