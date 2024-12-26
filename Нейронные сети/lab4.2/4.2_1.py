import numpy as np
import matplotlib.pyplot as plt

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)  # Производная сигмоиды
    return 1 / (1 + np.exp(-x))  # Сигмоида

# Входные данные
X = np.array([[2.1598, 2.1598, 2.1701, 2.1788],
              [2.1701, 2.1788, 2.1727, 2.1727],
              [2.1727, 2.1727, 2.1727, 2.1727],
              [2.1727, 2.1727, 2.1590, 2.1569],
              [2.1590, 2.1569, 2.1532, 2.1503]])

# Выходные данные
y = np.array([[2.1727],
              [2.1727],
              [2.1590],
              [2.1532],
              [2.1503]])

# Нормализация входных данных
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# Нормализация выходных данных
y_mean = np.mean(y)
y_std = np.std(y)
y_normalized = (y - y_mean) / y_std

np.random.seed(1)

# Инициализация весов
syn0 = 2 * np.random.random((4, 5)) - 1  # Входной слой (4 нейрона) к скрытому слою (5 нейронов)
syn1 = 2 * np.random.random((5, 3)) - 1  # Скрытый слой (5 нейронов) к следующему скрытому слою (3 нейрона)
syn2 = 2 * np.random.random((3, 1)) - 1    # Скрытый слой (3 нейрона) к выходному слою (1 нейрон)

# Обучение сети
errors = []  # Список для хранения ошибок

for j in range(60000):
    # Прямое распространение
    l0 = X_normalized
    l1 = nonlin(np.dot(l0, syn0))  # Первый скрытый слой
    l2 = nonlin(np.dot(l1, syn1))   # Второй скрытый слой
    l3 = nonlin(np.dot(l2, syn2))   # Выходной слой

    # Вычисление ошибки
    l3_error = y_normalized - l3
    errors.append(np.mean(np.abs(l3_error)))

    if (j % 10000) == 0:
        print("Error at iteration {}: {}".format(j, np.mean(np.abs(l3_error))))
        
    # Обратное распространение
    l3_delta = l3_error * nonlin(l3, deriv=True)  
    l2_error = l3_delta.dot(syn2.T) 
    l2_delta = l2_error * nonlin(l2, deriv=True)  
    l1_error = l2_delta.dot(syn1.T)  
    l1_delta = l1_error * nonlin(l1, deriv=True)  

    # Обновление весов
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)  
    syn0 += l0.T.dot(l1_delta)  

# Обратная нормализация результатов
predicted_normalized = l3  # Получаем предсказанные значения
predicted = predicted_normalized * y_std + y_mean  # Обратная нормализация


# Вывод весов
print(syn0)
print(syn1)
print(syn2)

# Вывод результатов
print("Выходные данные после тренировки (нормализованные):")
print(predicted_normalized)  # Нормализованные предсказания
print("Выходные данные после тренировки (обратная нормализация):")
print(predicted)  # Обратные нормализованные предсказания

# Визуализация ошибок
plt.plot(errors)
plt.title('Error during training')
plt.xlabel('Iterations (in thousands)')
plt.ylabel('Mean Absolute Error')
plt.show()