mass = []
sumAllEl = 0
countStr = 0
with open("C:/Users/nikit/Desktop/online_retail_II.txt", "r", encoding='utf-8') as file:
    for line in file:
        countStr += 1
        countEl = 8
        mass = line.split(",")
        for i in mass:
            if i == '':
                countEl -= 1
        sumAllEl += countEl
        # if count < 8:
        #     print(count, "|", line)
print("Общее кол-во эл в файле:", countStr * 8)
print("Кол-во пропущенных эл:", countStr * 8 - sumAllEl)
print("Процент пропущенных эл:", ((countStr * 8 - sumAllEl) / (countStr * 8)) * 100, "%")