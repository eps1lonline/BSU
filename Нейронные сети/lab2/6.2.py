str = "привет как дела я не робот"

alph = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
bAlph = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

list = str.split()

for el in list:
    for i in range(len(alph)):
        if el[0] == alph[i]:
            print(bAlph[i] + el[1:] + " ", end = "") 
