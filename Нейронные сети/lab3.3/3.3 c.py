# Список имен
surnames = ["Статкевич", "Слук", "Рудинский", "Налетько", "Канашевич", "Криворотов", "Лукин", "Козлов", "Бекетов", "Ёда"]
names = ["Захар", "Евгений", "Егор", "Арина", "Кирилл", "Евгений", "Антон", "Влад", "Дима", "Никита"]

# Запись фамилий с именами в новый файл
with open("full_names.txt", "w", encoding="utf-8") as file:
    for surname, name in zip(surnames, names):
        full_name = f"{surname.strip()} {name}"
        file.write(full_name + "\n")

# Чтение и печать файла
with open("full_names.txt", "r", encoding="utf-8") as file:
    print(file.read())