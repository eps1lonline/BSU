digit = 12
copyDigit = digit
newDigit = ""

for i in range(len(str(digit))):
    newDigit = newDigit + str(copyDigit % 10)
    copyDigit = copyDigit // 10

print(newDigit, " - ", digit, " = ", int(newDigit) - digit)