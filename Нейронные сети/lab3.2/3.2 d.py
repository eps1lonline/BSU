def max_custom(*args):
    if not args:
        raise ValueError("At least one argument is required.")
    maximum = args[0]
    for num in args:
        if num > maximum:
            maximum = num
    return maximum

print(max_custom(1, 3, 5, 7, 9))