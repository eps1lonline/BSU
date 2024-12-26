import numpy as np
import matplotlib.pyplot as plt

# Сигмоида
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# набор входных данных
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# выходные данные
y = np.array([[0, 0, 1, 1]]).T

# сделаем случайные числа более определёнными
np.random.seed(1)

# инициализируем веса случайным образом со средним 0
syn0 = 2 * np.random.random((3, 1)) - 1

# Список для хранения значений l1_error
errors = []

b = True

for iter in range(10000):
    # прямое распространение
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # насколько мы ошиблись?
    l1_error = y - l1
    errors.append(np.mean(np.abs(l1_error)))  # Сохраняем среднюю ошибку

    # перемножим это с наклоном сигмоиды
    l1_delta = l1_error * nonlin(l1, True)

    # обновим веса
    syn0 += np.dot(l0.T, l1_delta)

    if b:
        print("l1 после первой итерации:\n", l1)
        b = False

print("l1 после последней итерации:\n", l1)
print("Выходные данные после тренировки:\n", l1)
print("Веса:\n", syn0)

# Построение графика
plt.plot(errors)
plt.title('Средняя ошибка l1 на каждой итерации')
plt.xlabel('Итерации')
plt.ylabel('Средняя ошибка')
plt.show()