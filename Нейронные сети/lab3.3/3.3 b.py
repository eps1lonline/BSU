with open("surnames.txt", "r", encoding="utf-8") as file:
    surnames = file.readlines()

for i, surname in enumerate(surnames, start=1):
    print(f"{i}. {surname.strip()}")