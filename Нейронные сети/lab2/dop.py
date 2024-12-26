def count_divisors(num):
    divisors = []
    for i in range(1, int(num**0.5) + 1):
        if num % i == 0:
            divisors.append(i)
            if i != num // i:
                divisors.append(num // i)
    return divisors

def numbers_with_six_or_more_divisors(n):
    result = []
    for i in range(1, n + 1):
        divisors = count_divisors(i)
        if len(divisors) >= 6:
            result.append((i, divisors))
    return result

n = 30  # Задайте значение n
result = numbers_with_six_or_more_divisors(n)

for number, divisors in result:
    print(f"Число: {number}, Делители: {divisors}")