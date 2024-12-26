# Список фамилий
surnames = ["Статкевич", "Слук", "Рудинский", "Налетько", "Канашевич", "Криворотов", "Лукин", "Козлов", "Бекетов", "Ёда"]

# Запись в файл
with open("surnames.txt", "w", encoding="utf-8") as file:
    for surname in surnames:
        file.write(surname + "\n")

# Чтение и печать файла
with open("surnames.txt", "r", encoding="utf-8") as file:
    print(file.read())