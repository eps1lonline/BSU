def all_combinations(a):
    if len(a) <= 1:
        yield a
    else:
        head = ''
        tail = list(a)
        while len(tail) > 0:
            head += tail.pop(0)
            for s in all_combinations(tail):
                yield [head] + s

def all_signs(n):
    if n == 0:
        yield ()
    else:
        for tail in all_signs(n-1):
            for s in '+-':
                yield (s,) + tail

def perform_operations(nums, signs):
    nums = list(map(int, nums))
    result = nums.pop(0)
    n = 1
    for s in signs:
        if s == '+':
            result += nums.pop(0)
        if s == '-': 
            result -= nums.pop(0)
        n += 1
    return result

for numbers in all_combinations(tuple(map(str, range(2, 7)))):
    #print(numbers)
    for signs in all_signs(len(numbers) - 1):
        #print(signs)
        summ = perform_operations(numbers, signs)
        if summ == 50:
            print(
                ''.join(map(
                    lambda x: ''.join(x),
                    zip(numbers, signs)))
                + numbers[-1])