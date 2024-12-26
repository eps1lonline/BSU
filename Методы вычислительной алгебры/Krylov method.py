import numpy as np
from sympy import Symbol, solve

A = [[0.3857, -0.0508, 0.0102, 0.0203, 0.0711],
    [0.0528, 0.6039, 0.0000, -0.0406, 0.0406],
    [0.0305, 0.0000, 0.4852, -0.1421, 0.0812],
    [-0.0609, 0.1279, 0.0000, 0.4711, -0.0203],
    [0.2538, 0.0000, 0.0914, 0.0102, 0.5684]]

n = 5
a = np.array(A)
At = a.transpose()
print("Симметрическая A*A^T:\n\t", At)
a = np.dot(At, a)

c = []
c.append([1, 0, 0, 0, 0]) 
for i in range(1, n + 1):
    c.append(np.dot(a, c[i - 1])) 

print("\nВекторы C^i:")
for el in c:
    print("\t", el)

C = np.array(c) 
cn = c.pop() 
c = np.array(c).transpose()
for i in range(n):
    c[i] = list(reversed(c[i]))
p = np.linalg.solve(c, cn)

print("\nКоэффициенты собственного многочлена P(lambda):\n\t", p) 

x = Symbol('x')
Lambda = solve(x**5 - p[0] * x**4 - p[1] * x**3 - p[2] * x**2 - p[3] * x - p[4], x)
print("\nСобственные значения lambda:\n\t", Lambda)

l = max(Lambda)
print("\nМаксимальное собственное lambdaMax:\n\t", l)

b = np.ones(n)
for i in range(1, n):
    b[i] = b[i - 1] * l - p[i - 1]
print("\nКоэффиценты B^i:\n\t", b)

x = np.sum([b[i] * C[n - i - 1] for i in range(n)], axis=0)
print("\nСобственный вектор матрицы A - x(lambdaMax):\n\t", x)

r = np.dot(a, x) - l * x 
print("\nВектор невязки r:\n\t", r)

rnorm = np.linalg.norm(r, 1)
print("\nНорма невязки ||r||:\n\t", rnorm)