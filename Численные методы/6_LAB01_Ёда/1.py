def f(x):
    return x**3 + 3*x + 1

def phi(x):
    return (x**3 + 1) / 3

def simple_iteration_method(x0, epsilon=1e-7, max_iter=1000):
    iteration = 0

    while True:
        x_next = phi(x0)

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
solution, iterations, residual = simple_iteration_method(initial_guess)

print("Приближенное решение:", solution)
print("Количество итераций:", iterations)
print("Невязка:", residual)