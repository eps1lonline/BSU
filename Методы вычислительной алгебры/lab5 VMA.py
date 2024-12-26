import numpy as np
from sympy import symbols, solve

A = [[6, -2, 2],
     [-2, 5, 0],
     [2, 0, 7]]

a = np.array(A)
n = 3
print("Матрица A:\n\t", A)

At = a.transpose()
print(At)

a = np.dot(At, a)
print("\nСимметрическая матрица A*A^T:\n\t", a)

f = a
s = np.identity(n)

for i in range(n - 1):
    m = np.identity(n)

    m[n - 2 - i] = f[n - 1 - i]
    print("b\n\t", m)
    f = np.dot(m, f)
    f = np.dot(f, np.linalg.inv(m))
    s = np.dot(s, np.linalg.inv(m))
    print("d\n\t", s)

print("\nКаноническая Ф:\n\t", f)
print("\nМатрица мреобразования S:\n\t", s)

p = f[0]
print("\nКоэффициенты собственного многочлена P(lambda):\n\t", p)

x = symbols('x') 
Lambda = solve(x**3 - p[0] * x**2 - p[1] * x - p[2], x)
# Lambda = solve(x**5 - p[0] * x**4 - p[1] * x**3 - p[2] * x**2 - p[3] * x - p[4], x)
print("\nСобственные значения lambda:\n\t", Lambda)

maxLambda = max(Lambda)
print("\nmax(lambda):\n\t", maxLambda)

y = [maxLambda ** i for i in range(n - 1, -1, -1)]
print("\nСобственный вектор матрицы Ф — y(max(lambda))∶\n\t", y)

x = np.dot(s, y)
print("\nСобственный вектор матрицы А — x(max(lambda))∶\n\t", x)

r = np.dot(a, x) - maxLambda * x
print("\nВектор невязки r:\n\t", r)
