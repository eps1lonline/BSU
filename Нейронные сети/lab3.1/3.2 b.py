mass = [1, 2, 3, 4, 5]

youDigit = 2

flag = False
for el in mass:
    if youDigit == el:
        flag = True

if flag == True:
    print("YES")
else:
    print("NO")