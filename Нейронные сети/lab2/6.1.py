str = "привет как дела я не робот"

glas = "аеёиоуыэюя"
glassCount = 0

soglas = "бвгджзйклмнпрстфхцчшщ"
soglasCount = 0

for i in range(len(str)):
    for j in range(len(glas)):
        if str[i] == glas[j]:
            glassCount += 1

    for j in range(len(soglas)):
        if str[i] == soglas[j]:
            soglasCount += 1

print("Гласные: ", glassCount, "\nСогласные: ", soglasCount)