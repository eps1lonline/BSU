import numpy as np
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.datasets import imdb

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)

# Максимальное количество слов (по частоте использования)
max_features = 5000
# Максимальная длина рецензии в словах
maxlen = 80

# Загружаем модель из файла
with open("cc_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("cc_model.weights.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Загрузили Model")

# Загружаем текст и преобразуем его в формат, подходящий для модели
with open('text.txt', 'r') as file:
    text_data = file.read()

# Преобразуем текст в one-hot представление
x = text.one_hot(text_data, max_features)

# Паддинг, чтобы длина входа соответствовала maxlen
x = sequence.pad_sequences([x], maxlen=maxlen)

# Запускаем распознавание объекта
prediction = loaded_model.predict(x)
print(prediction)

if (prediction < 0.94):
    print("Отзыв отрицательный")
else:
    print("Отзыв положительный")