list = [33, 11, 44, 123, 6, 23, 43, 6, 213]

min = list[0]
max = list[0]

for el in list:
    if el >= max:
        max = el
    if el <= min:
        min = el

print("min: ", min, "\nmax: ", max)

countMin = 0
countMax = 0

for el in list:
    if el == min:
        countMin += 1
    if el == max:
        countMax += 1

print("countMin: ", countMin, "\ncountMax: ", countMax)