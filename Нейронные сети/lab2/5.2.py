list = [33, 11, 44, 123, 6, 23, 43, 6, 213]
i = 1
for i in range(9):
    if list[i] >= list[i-1]:
        print(list[i])