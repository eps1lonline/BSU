import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import unittest

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)  # Производная сигмоиды
    return 1 / (1 + np.exp(-x))  # Сигмоида

def train_neural_network(texts, y):
    # Токенизация
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform([text.split() for text in texts])

    np.random.seed(1)

    # Инициализация весов
    syn0 = 2 * np.random.random((X.shape[1], 4)) - 1  # Входной слой
    syn1 = 2 * np.random.random((4, 2)) - 1   # Скрытый слой к выходному слою

    # Обучение сети
    for j in range(60000):
        # Прямое распространение
        l0 = X
        l1 = nonlin(np.dot(l0, syn0))  # Первый скрытый слой
        l2 = nonlin(np.dot(l1, syn1))   # Выходной слой

        # Вычисление ошибки
        l2_error = y - l2
        
        if (j % 10000) == 0:
            print("Error:" + str(np.mean(np.abs(l2_error))))
            
        # Обратное распространение
        l2_delta = l2_error * nonlin(l2, deriv=True)
        l1_error = l2_delta.dot(syn1.T) 
        l1_delta = l1_error * nonlin(l1, deriv=True)  

        # Обновление весов
        syn1 += l1.T.dot(l2_delta)  
        syn0 += l0.T.dot(l1_delta)  

    return syn0, syn1, mlb  # Возвращаем веса и MultiLabelBinarizer

def predict(texts, syn0, syn1, mlb):
    X = mlb.transform([text.split() for text in texts])  # Используем тот же mlb для тестирования
    
    l1 = nonlin(np.dot(X, syn0))  # Первый скрытый слой
    l2 = nonlin(np.dot(l1, syn1))  # Выходной слой
    return l2  # Возвращаем выходные данные

# Обучающие данные
texts = [
    "купить лекарства",        # спам
    "лучшие цены на товары",   # спам
    "срочно вернуть деньги",    # спам
    "встреча завтра",          # не спам
    "как дела",                # не спам
    "погода сегодня хорошая",   # не спам
    "скидки на обувь",         # спам
    "новости спорта",          # не спам
]

y = np.array([[1, 0],  # спам
              [1, 0],  # спам
              [1, 0],  # спам
              [0, 1],  # не спам
              [0, 1],  # не спам
              [0, 1],  # не спам
              [1, 0],  # спам
              [0, 1]]) # не спам

# Обучение нейросети
syn0, syn1, mlb = train_neural_network(texts, y)

# Тестовые данные
test_texts = [
    "купить телефон",           # спам
    "скидки в мегатоп",         # спам
    "привет",                # не спам
    "как твои дела",        # не спам
    "как дела",              # не спам
    "погода сегодня хорошая", # не спам
    "скидки на обувь",       # спам
    "новости спорта",        # не спам
]

# Прогнозирование
l2_test = predict(test_texts, syn0, syn1, mlb)

# Вывод результатов
print("Результаты для тестовых случаев:")
for i, text in enumerate(test_texts):
    print(f"Текст: '{text}', Вероятность спама: {l2_test[i][0]:.3f}, Вероятность не спама: {l2_test[i][1]:.3f}")