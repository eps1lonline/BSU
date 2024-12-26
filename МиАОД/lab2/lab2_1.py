def identify_scale(value):
    if isinstance(value, str):
        return "Номинальная"
    elif isinstance(value, int):
        return "Порядковая"
    elif isinstance(value, float):
        return "Интервальная"
    else:
        return "Неизвестно"

data = [
    "мужчина",  # пол
    25,         # возраст
    "высшее",   # уровень образования
    55000.50    # зарплата
]

scales = [identify_scale(item) for item in data]

for item, scale in zip(data, scales):
    print(f"Значение: {item}, Шкала: {scale}")