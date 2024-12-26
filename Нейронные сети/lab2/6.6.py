list = ["Привет", "как", "дела", "я", "не", "робот"]

ptr = list[0]

for i in range(int(len(list))):
    if i == int(len(list)) - 1:
        list[i] = ptr
    else:
        list[i] = list[i + 1]

print(list)
