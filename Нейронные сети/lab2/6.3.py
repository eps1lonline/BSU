str = "привет как дела я не робот"

str = str.replace(" ", "")

list = {}

for letter in str:
    if letter in list:
        list[letter] += 1
    else:
        list[letter] = 1

for letter, freq in list.items():
    print(f"Буква '{letter}' встречается {freq} раз(а)")