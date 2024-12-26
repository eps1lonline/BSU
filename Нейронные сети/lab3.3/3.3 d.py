with open("full_names.txt", "r", encoding="utf-8") as file:
    full_names = file.readlines()

for full_name in full_names:
    surname, name = full_name.strip().split()
    print(f"{surname} {name[0]}.")