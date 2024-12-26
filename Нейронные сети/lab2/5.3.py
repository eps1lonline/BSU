list = [33, 11, 44, 123, 6, 23, 43, 6, 213]
max = list[0]
for i in range(9):
    if list[i] >= max:
        max = list[i]
print(max)