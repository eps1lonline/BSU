import math

# Координаты дверей
doors = [
    (2.5, 0.5),
    (3.5, 0.5),
    (9, 1.5),
    (1, 2.5),
    (5, 2.5),
    (9, 2.5),
    (1, 7.5),
    (5, 7.5),
    (9, 5.5),
    (1, 8.5),
    (5, 8.5),
    (7.5, 9.5),
    (11.5, 6.5),
    (11.5, 3.5)
]

# Функция для вычисления евклидова расстояния
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Суммирование расстояний
distance_sums = []
for i in range(len(doors)):
    total_distance = 0
    for j in range(len(doors)):
        if i != j:  # Не учитывать расстояние до самой себя
            total_distance += euclidean_distance(doors[i], doors[j])
    distance_sums.append(total_distance)

# Вывод результата
for i, total in enumerate(distance_sums):
    print(f"Сумма расстояний от двери {i + 1}: {total:.2f}")