def power(a, n):
    result = 1
    for _ in range(abs(n)):
        result *= a
    return result if n >= 0 else 1 / result

a = float(input("Enter a: "))
n = int(input("Enter n: "))
print("a^n:", power(a, n))