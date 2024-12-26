import numpy as np

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)  # Производная сигмоиды
    return 1 / (1 + np.exp(-x))  # Сигмоида

# Входные данные
X = np.array([[0.645, 33, 4, 0.34],
              [2.880, 152, 14, 1.5],
              [1.110, 88, 5, 0.475],
              [10.400, 260, 3, 6],
              [2.880, 152, 3, 1.5],
              [9.895, 122, 30, 6.2],
              [3.695, 136, 3, 3],
              [2.170, 116, 5, 0.6],
              [27.895, 278, 150, 11]])

# Нормализация данных для каждого столбца
def normalize_column(column, thresholds):
    normalized = []
    for value in column:
        if value <= thresholds[0]:
            normalized.append(0)
        elif thresholds[0] < value <= thresholds[1]:
            normalized.append(0.25)
        elif thresholds[1] < value <= thresholds[2]:
            normalized.append(0.5)
        elif thresholds[2] < value <= thresholds[3]:
            normalized.append(0.75)
        else:
            normalized.append(1)
    return np.array(normalized)

# Определение порогов для каждого столбца
thresholds_list = [
    [0.5, 1, 2, 5],    # Для первого столбца (масса)
    [20, 50, 100, 200], # Для второго столбца (мощность)
    [2, 5, 10, 20],     # Для третьего столбца (пассажировместимость)
    [1, 2, 3, 4]        # Для четвертого столбца (грузоподъемность)
]

# Нормализация каждого столбца
X_normalized = np.column_stack([
    normalize_column(X[:, i], thresholds_list[i]) for i in range(X.shape[1])
])

# Выходные данные ЛЕГКОВОЙ ПАССАЖИРСКИЙ ГРУЗОВОЙ
y = np.array([[1, 0, 0],
              [0, 1, 0],
              [1, 0, 0],
              [0, 0, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]])

np.random.seed(1)

# Инициализация весов
syn0 = 2 * np.random.random((4, 4)) - 1  # Входной слой (4 нейрона) к скрытому слою (4 нейрона)
syn1 = 2 * np.random.random((4, 3)) - 1  # Скрытый слой (4 нейрона) к выходному слою (3 нейрона)

# Обучение сети
for j in range(60000):
    # Прямое распространение
    l0 = X_normalized
    l1 = nonlin(np.dot(l0, syn0))  # Первый скрытый слой
    l2 = nonlin(np.dot(l1, syn1))   # Выходной слой

    # Вычисление ошибки
    l2_error = y - l2
    
    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
        
    # Обратное распространение
    l2_delta = l2_error * nonlin(l2, deriv=True)  # Ошибка на выходе
    l1_error = l2_delta.dot(syn1.T)  # Ошибка для первого скрытого слоя
    l1_delta = l1_error * nonlin(l1, deriv=True)  # Ошибка с учетом производной

    # Обновление весов
    syn1 += l1.T.dot(l2_delta)  # Обновление весов между скрытым и выходным слоями
    syn0 += l0.T.dot(l1_delta)  # Обновление весов между входным и скрытым слоями

# Вывод результатов с округлением
print("Выходные данные после тренировки:")
print(np.round(l2, 3))





# проверка 

def predict_transport(features):
    # Нормализуем входные данные
    normalized_features = np.array([normalize_column(np.array([feature]), thresholds_list[i]) for i, feature in enumerate(features)]).flatten()
    # Прямое распространение
    l1 = nonlin(np.dot(normalized_features, syn0))  # Первый скрытый слой
    l2 = nonlin(np.dot(l1, syn1))   # Выходной слой
    return np.round(l2, 3)

# Пример использования: предсказание типа транспорта
new_data1 = [74, 1050, 1, 90]
new_data2 = [13.6, 260, 10, 3]
new_data3 = [0.3, 40, 3, 0.25]
new_data4 = [0.2, 1, 4, 0.2]
new_data5 = [0.02, 0.3, 1, 0.1]
new_data6 = [18.4, 245, 168, 12]
predicted_output = predict_transport(new_data1)
print("Предсказанный тип транспорта для Белаз:", predicted_output)
predicted_output = predict_transport(new_data2)
print("Предсказанный тип транспорта БТР80:", predicted_output)
predicted_output = predict_transport(new_data3)
print("Предсказанный тип транспорта Урал мото:", predicted_output)
predicted_output = predict_transport(new_data4)
print("Предсказанный тип транспорта Повозка:", predicted_output)
predicted_output = predict_transport(new_data5)
print("Предсказанный тип транспорта Велосипед:", predicted_output)
predicted_output = predict_transport(new_data6)
print("Предсказанный тип транспорта Трамвай:", predicted_output)