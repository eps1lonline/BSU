import os
import cv2
import time
import math
import shutil
import matplotlib.pyplot as plt

from ultralytics import YOLO, solutions

# Запуск таймера (узнаем сколько по времени работает наша программа)
start_time = time.time()

# Путь к main папке
main_patch = "C:/Users/nikit/Desktop/Diplom"

# Словарь папок, которые будем создавать
directories = {
    "graphic":              os.path.join(main_patch, "graphic"),
    "final_video":          os.path.join(main_patch, "final_video"),
    "time_series":          os.path.join(main_patch, "time_series"),
    "screenshots":          os.path.join(main_patch, "screenshots"),
    "screenshots_anomaly":  os.path.join(main_patch, "screenshots_anomaly")
}

def delete_directory(dir):
    """Удаляет директории"""

    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)
        print(f"Папка '{dir}' удалена")
    else:
        print(f"Папка '{dir}' не найдена")

# Удаляем диркетории
for dir in directories.values():
    delete_directory(dir)
print()

def create_directory(dir):
    """Создает директорию"""

    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Папка '{dir}' создана")
    else:
        print(f"Папка '{dir}' не найдена")

# Создаём директории
for dir in directories.values():
    create_directory(dir)
print()





# Коэффициенты для вычисления расстояния
coef_1 = 10 # people
# coef_1 = 2 # car_1
# coef_1 = 1 # car_2

# Коэффициенты для обнаружения аномалий
coef_2 = 2 # people
# coef_2 = 20 # car_1
# coef_2 = 20 # car_2

# Открытие видеофайла
data_set = os.path.join(main_patch, "data_set/people.mp4")
# data_set = os.path.join(main_patch, "data_set/car_1.mp4")
# data_set = os.path.join(main_patch, "data_set/car_2.mp4")
cap = cv2.VideoCapture(data_set)

# Проверка на сущесвование файла
assert cap.isOpened(), "Ошибка чтения видеофайла"

# Получение параметров видео
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH, 
    cv2.CAP_PROP_FRAME_HEIGHT, 
    cv2.CAP_PROP_FPS
))

# Загрузка модели
model = YOLO("yolov8n.pt") 
names = model.model.names   

# Инициализация видеописателя
video_writer = cv2.VideoWriter(
    os.path.join(directories["final_video"], "final_video.avi"), 
    cv2.VideoWriter_fourcc(*"mp4v"), 
    fps, 
    (w, h)
)

# Определение горизонтальной линии (розовая линия)
line_pts = [(0, int(h/2)), (int(w), int(h/2))]

# Инициализация объекта оценки скорости
speed_obj = solutions.SpeedEstimator(reg_pts=line_pts, names=names, view_img=True)





# Переменные для сохранения скриншотов
count = 0
flag = False

# Сохраняю каждую координату объекта и его время в каждый момент
all_distance = {}
all_time = {}

# Прогресс программы
progress = 1

# Ядро скрипта обрабатывает видео кадр за кадром
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    # Отслеживание объектов и оценка скорости
    tracks = model.track(im0, persist=True, show=False, verbose=False)
    im0 = speed_obj.estimate_speed(im0, tracks, all_time, all_distance, coef_1)
    video_writer.write(im0)

    # Создание скриншотов автомобилей при пересечении линии
    if len(speed_obj.trkd_ids) != count and flag:
        for i in range(count, len(speed_obj.trkd_ids)):
            cv2.imwrite(os.path.join(directories["screenshots"], f"screenshot_{speed_obj.trkd_ids[i]}.png"), im0)
            print(f"'screenshot_{speed_obj.trkd_ids[i]}.png' сохранён")
            count += 1
        flag = False

    if len(speed_obj.trkd_ids) != count:
        flag = True
    
    # Сбор данных о расстоянии
    for i in speed_obj.trk_history:
        if i not in all_distance:
            all_distance[i] = []
        length = len(speed_obj.trk_history[i]) - 1
        all_distance[i].append(speed_obj.trk_history[i][length])

    # Сбор данных о времени
    for i in speed_obj.trk_pt:
        if i not in all_time:
            all_time[i] = []
        all_time[i].append(speed_obj.trk_pt[i])

    print(f'Progress: ({progress}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})')
    progress += 1

# Освобождение ресурсов
cap.release() 
video_writer.release()
cv2.destroyAllWindows()

print(f"\nВидео обработано и сохранено '{directories['final_video']}'")





# Убирает повторы в словаре all_distance
for i in all_distance:
    seen = set()
    all_distance[i] = [j for j in all_distance[i] if not (j in seen or seen.add(j))]

# Убирает повторы в словаре all_time
for i in all_time:
    seen = set()
    all_time[i] = [j for j in all_time[i] if not (j in seen or seen.add(j))]
    
def distance_function(all_distance, i, j, d):
    """Находит расстояние от одной точки до другой"""

    x1 = all_distance[i][j][0]
    y1 = all_distance[i][j][1]
    x2 = all_distance[i][j + 1][0]
    y2 = all_distance[i][j + 1][1]
    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    return d

# Запись временного ряда в файл
with open(os.path.join(directories["time_series"], "time_series.txt"), 'w') as file:
    file.write(f'{"id":<10} {"coordinates(x;y)":<50} {"time(s)":<25} {"distance(m)":<25} {"speed(km/h)":<25}\n')
    
    for i in speed_obj.trkd_ids:
        d = 0
       
        for j in range(0, len(all_distance[i])):
            # Расстояние
            if j != 0:
                d += distance_function(all_distance, i, j - 1, d)

            # Время
            t = all_time[i][j] - all_time[i][0]

            # Скорость
            if t != 0:
                u = ((d / coef_1) / t) * 3.6
            else:
                u = 0

            # Вывод
            x, y = all_distance[i][j]
            file.write(f'{i:<10} {f"({x}, {y})":<50} {t:<25} {(d / coef_1):<25} {(u):<25}\n')
        file.write('\n')

print(f"Временной ряд сформирован и сохранён '{directories['time_series']}'")





# Переменные для рисования графика
speed_list = list(speed_obj.spd.values())
average_speed = sum(speed_list) / len(speed_list)
max_x = max(speed_obj.trkd_ids)
max_y = max(speed_list)

# Рисуем сам график
plt.figure(figsize=(16, 9))  # Размер графика на весь экран
plt.title('График скоростей', fontsize=20, fontname='Times New Roman') # Название графика

plt.axis([0, max_x, 0, max_y]) # Размерность по осям
plt.xticks(speed_obj.trkd_ids) # На оси Х только точки из trkd_ids
# plt.yticks(speed_list) # На оси Y только точки из trkd_ids

plt.xlabel('№ объекта', color='gray') # Название оси Х
plt.ylabel('Скорость', color='gray') # Название оси Y
plt.grid(True) # Сетка

# Горизонтальные линии
plt.plot([0, max_x], [average_speed, average_speed], color='green', linestyle='--') 
plt.plot([0, max_x], [average_speed - coef_2, average_speed - coef_2], color='red', linestyle='--')
plt.plot([0, max_x], [average_speed + coef_2, average_speed + coef_2], color='orange', linestyle='--')

# Точки на графике
for i in speed_obj.trkd_ids:
    if speed_obj.spd[i] <= average_speed - coef_2 or speed_obj.spd[i] >= average_speed + coef_2:
        plt.plot([i], [speed_obj.spd[i]], color='red', marker='D')
        plt.plot([i, i], [0, speed_obj.spd[i]], color='black')
        plt.text(i, speed_obj.spd[i], 'Аномалия')

        # Сохраняем скриншот аномалии
        shutil.copy(os.path.join(directories["screenshots"], f"screenshot_{i}.png"), directories["screenshots_anomaly"])

    else:
        plt.plot([i], [speed_obj.spd[i]], color='green', marker='D')
        plt.plot([i, i], [0, speed_obj.spd[i]], color='black')

plt.legend(['Ср. скорость', f'Ср. скорость - {coef_2}', f'Ср. скорость + {coef_2}'], loc=2) # Легенда
plt.savefig(os.path.join(directories["graphic"], "graphic.png")) # Сохранение графика
plt.show() # Показать график

print(f"График аномальных явлений построен и сохранён '{directories['graphic']}'")





# Время работы программы
end_time = time.time()
execution_time = end_time - start_time
print(f"Время выполнения программы {execution_time:.2f} секунд")


