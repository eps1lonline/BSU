import itertools

# Все возможные операции между цифрами
operations = ['+', '-', '']

# Все возможные комбинации операций для четырёх промежутков
operation_combinations = list(itertools.product(operations, repeat=4))

# Список цифр
digits = ['1', '2', '3', '4', '5']

# Функция для генерации выражения из цифр и операций
def generate_expression(ops):
    expression = digits[0]
    for i in range(1, len(digits)):
        expression += ops[i-1] + digits[i]
    return expression

# Найдем все выражения, равные 50
results = []
for ops in operation_combinations:
    expression = generate_expression(ops)
    if eval(expression) == 50:
        results.append(expression)

results
