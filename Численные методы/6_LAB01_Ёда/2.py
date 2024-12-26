def f(x):
    return x**3 + 3*x + 1

def f_prime(x):
    return 3*x**2 + 3

def newton_method(x0, epsilon=1e-7, max_iter=1000):
    iteration = 0

    while True:
        x_next = x0 - f(x0) / f_prime(x0)

        if abs(x_next - x0) <= epsilon:
            break

        x0 = x_next
        iteration += 1

        if iteration >= max_iter:
            print("Максимальное число итераций достигнуто.")
            break

    residual = f(x_next) 
    return x_next, iteration, residual

initial_guess = -0.25 
solution, iterations, residual = newton_method(initial_guess)

print("Приближенное решение:", solution)
print("Количество итераций:", iterations)
print("Невязка:", residual)