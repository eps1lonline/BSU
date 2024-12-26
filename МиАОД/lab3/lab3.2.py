import re

def check_date_format(date_str):
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(pattern, date_str))

mass = []
countInvalidForm = 0
countStr = 0
print("Парвильный формат для даты: 'XXXX-XX-XX'")
with open("C:/Users/nikit/Desktop/GlobalLandTemperaturesByCity.txt", "r", encoding='utf-8') as file:
    for line in file:
        countStr += 1
        mass = line.split(",")
        if check_date_format(mass[0]) == False and countStr != 1:
            print(mass[0], "имеет неверный формат")
            countInvalidForm += 1
print("\nКол-во дат:", countStr)
print("Кол-во дат с правильным форматом:", countStr - countInvalidForm)
print("Кол-во дат с неправильным форматом:", countInvalidForm)