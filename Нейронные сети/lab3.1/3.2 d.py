import random
from random import random, randrange, randint

def funct(mass):
    for el in mass:
        if el == digit:
            return mass

n = 100
#digit = randint(0, n)
digit = 10
print("Загаданное число:\t", digit)

mass1 = [1, 2, 3, 4] 
mass2 = [3, 2, 10, 11, 22, 33]
mass3 = [1, 10]
mass4 = [3, 4, 5, 6]

trueMass = []

trueMass.append(funct(mass1))
trueMass.append(funct(mass2))
trueMass.append(funct(mass3))
trueMass.append(funct(mass4))

filtered_list = list(filter(lambda x: x is not None, trueMass))

for el in filtered_list:
    for j in el:
        print(j, " ", end="")
