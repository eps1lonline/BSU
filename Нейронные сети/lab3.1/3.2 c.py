N = 3
M = 2 

nLine = [2, 1, 0]
mLine = [1, 3]

together = []
onlyN = []
onlyM = []

for el1 in nLine:
    flag = False

    for el2 in mLine:
        if el1 == el2:
            together.append(el1)
            flag = True

    if flag == False:
        onlyN.append(el1)

for el1 in mLine:
    flag = False

    for el2 in nLine:
        if el1 == el2:
            flag = True

    if flag == False:
        onlyM.append(el1)

print("\nnLine - Аня\nmLine - Боря")

print("\nОбщие эл.:\t", together)
print("Эл. которые есть только у Ани:\t", onlyN)
print("Эл. которые есть только у Бори:\t", onlyM)

print("\nКол-во эл. у Ани:\t", len(nLine))
print("Кол-во эл. у Бори:\t", len(mLine))

nLine.sort()
mLine.sort()
print("\nСортировка эл. у Ани:\t", nLine)
print("Сортировка эл. у Бори:\t", mLine)