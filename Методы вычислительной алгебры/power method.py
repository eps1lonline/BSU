import numpy as np
from sympy import Symbol, solve

A = [[0.3857, -0.0508, 0.0102, 0.0203, 0.0711],
    [0.0528, 0.6039, 0.0000, -0.0406, 0.0406],
    [0.0305, 0.0000, 0.4852, -0.1421, 0.0812],
    [-0.0609, 0.1279, 0.0000, 0.4711, -0.0203],
    [0.2538, 0.0000, 0.0914, 0.0102, 0.5684]]

n = 5
eps = 10 ** (-15)
a = np.array(A)
At = a.transpose() 
print("Симметрическая A*A^T:\n\t", At)
a = np.dot(At, a) 

yk = np.ones(5) 
y = np.dot(a, yk) 
l = y[0] / yk[0] 
k = 1

while (True): 
    yk = np.dot(a, y)
    lk = yk[0] / y[0]
    yk /= max(yk)

    if abs(lk - l) <= eps:
        break

    y = yk
    l = lk
    k += 1

p = [1.42919690e+00, -7.58323068e-01, 1.84357781e-01, -2.02232134e-02, 7.83880398e-04]
print("\nКоэффициенты собственного многочлена P(lambda):\n\t", p)

x = Symbol('x')
Lambda = solve(x**5 - p[0] * x**4 - p[1] * x**3 - p[2] * x**2 - p[3] * x - p[4], x)
print("\nСобственные значения lambda:\n\t", Lambda)

l = max(Lambda)
print("\nМаксимальное собственное lambdaMax:\n\t" , l)

print("\nКоличество итераций k:\n\t", k)

r = np.dot(a, yk) - lk * yk
print("\nВектор невязки r:\n\t", r)

print("\nЭпсилон E:\n\t", eps)

p.insert(0, -1)
r1 = sum(-(lk ** (n - i)) * p[i] for i in range(n + 1))
print("\nНевязка Pn(lambda^k):\n\t", r1)

rnorm = np.linalg.norm(r, 1)
print("\nНорма невязки ||r||:\n\t", rnorm)

R = [[0.3857, -0.0508, 0.0102, 0.0203, 0.0711],
[0,              0.6109,         -0.0014,        -0.0434,        0.0309],
[0,              0,              0.4844,         -0.1434,        0.0754],
[0,              0,              0,              0.4834,         -0.0154],
[0,             0,              0,              0,             0.5075]]

rnorm = np.linalg.norm(R, 1)
print("\nНорма невязки ||r||:\n\t", rnorm)