from collections import Counter

import numpy as np
import scipy.stats as stats
import seaborn as sns # pip install seaborn
import statsmodels.api as sm # pip install statsmodels
import matplotlib.pyplot as plt # pip install matplotlib

def openFile(fileName, variant):
    mass = []
    countEl = 0
    with open(fileName, "r") as file:
        for line in file:
            ptr = line.split(",")
            if (ptr[variant] != "0" and countEl != 0):
                mass.append(float(ptr[variant]))
            countEl += 1
    print("Всего эл.:", countEl)
    print("Эл. в массиве:", len(mass))
    print("Удалено эл.:", countEl - len(mass))
    print("Массив:", mass[:20])
    return mass

# 1
variant = 6
fileName = "C:/Users/nikit/Desktop/женя про/avianHabitat.txt"
data1 = openFile(fileName, variant + 7)

# 2.1
print("Макс. эл.:", np.max(data1))
print("Мин. эл:", np.min(data1))

# 2.2m
print("Размах распределения:", np.max(data1) - np.min(data1))

# 2.3
print("Среднее значение:", np.mean(data1))

# 2.4
print("Медиана:", np.median(data1))

# 2.5
print("Мода:", Counter(data1))
    
# 2.6
print("Дисперсия:", np.var(data1))

# 2.7
print("Среднеквадратическое отклонение:", np.sqrt(np.var(data1)))

# 2.8
print("Первый квартиль:", np.percentile(data1, 25))
print("Третий квартиль:", np.percentile(data1, 75))

# 2.9
print("Интерквартильный размах:", np.percentile(data1, 75) - np.percentile(data1, 25))

# 2.10
print("Асимметрия данных:", (len(data1) / ((len(data1) - 1) * (len(data1) - 2))) * np.sum(((data1 - np.mean(data1)) / np.std(data1)) ** 3))

# 2.11
print("Эксцесс данных:", np.mean((data1 - np.mean(data1))**4) / np.var(data1)**2 - 3)

# 3
plt.figure(figsize=(8, 6))
sns.boxplot(data=data1)
plt.title('Диаграмма с усами')
plt.xlabel('Данные')
plt.ylabel('Значения')
plt.show()

# 4
if (variant == 6):
    variant = 5
elif (variant == 1):
    variant = 2
else:
    variant += 1
data2 = openFile(fileName, variant)
plt.figure(figsize=(10, 6))
sns.boxplot(data=[data1, data2]) 
plt.title('Диаграммы с усами для двух наборов данных')
plt.xlabel('Наборы данных')
plt.ylabel('Значения')
plt.xticks([0, 1], ['Мой вариант', 'Соседний вариант'])
plt.show()

# 5
# собственная реализация
data1Sort = np.sort(data1)
yValues = np.arange(1, len(data1Sort) + 1) / len(data1Sort)
plt.step(data1Sort, yValues, label='ЭФР (собственная)', where='post')
# statsmodels
ecdf = sm.distributions.ECDF(data1)
data1Sort = data1Sort - 0.3 # для того, чтобы увидеть разницу, т.к. графики идентичны 
plt.step(data1Sort, ecdf(data1Sort), label='ЭФР (statsmodels)', where='post')
plt.title('Эмпирическая функция распределения')
plt.xlabel('Значения')
plt.ylabel('Вероятность')
plt.legend()
plt.grid(True)
plt.show()

# 6
sns.histplot(data1, bins=10, kde=True, stat='probability')
plt.title('Гистограмма вероятностей и сглаженная кривая')
plt.xlabel('Значения')
plt.ylabel('Вероятность')
plt.grid(True)
plt.show()

# 7
stats.probplot(data1, dist="norm", plot=plt)
plt.title('QQ-график для проверки нормальности')
plt.grid(True)
plt.show()