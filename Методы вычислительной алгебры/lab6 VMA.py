import numpy as np
import math

A = [[6, -2, 2], [-2, 5, 0], [2, 0, 7]]

a = np.array(A)
n = 3
print("Матрица A:\n\t", A)

At = a.transpose()

a = np.dot(At, a)
print("\nСимметрическая матрица A*A^T:\n\t", a)

E = np.identity(n)
U = np.identity(n)

eps = 0.00001

k = 0
ak = a

while (True):
    L = np.tril(ak)
    temp = np.absolute(ak - L)

    sigma = sum([abs(el) ** 2 for el in temp])
    if np.all(sigma <= eps):
        break

    i, j = np.unravel_index(temp.argmax(), temp.shape)
    alpha = math.atan(2 * ak[i][j] / (ak[i][i] - ak[j][j])) / 2
    uk = np.identity(n)
    uk[i][i] = math.cos(alpha)
    uk[i][j] = -math.sin(alpha)
    uk[j][j] = math.cos(alpha)
    uk[j][i] = math.sin(alpha)
    
    ak = np.dot(np.dot(uk.transpose(), ak), uk)
    U = np.dot(U, uk)
    k += 1

print("\nМатрица Λ=U^T*AU:\n\t", ak)
# p = [1.42919690e+00, -7.58323068e-01, 1.84357781e-01, -2.02232134e-02, 7.83880398e-04]
p = [162, -99, 18]
print("\nКоэффициенты собственного многочлена P(lambda):\n\t", p)

maxLambda = max(ak.diagonal())
print("\nmax(lambda):\n\t", maxLambda)

print("\nМатрица U:\n\t", U)

x = U.transpose()[ak.diagonal().argmax()]
x /= max(x)
print("\nСобственный вектор матрицы А — x(max(lambda))∶\n\t", x)

print("\nКол-во:\n\t", k)
print("\nEpsilon:\n\t", eps)

r = np.dot(a, x) - maxLambda * x 
print("\nВектор невязки r:\n\t", r)

p.insert(0, -1)
r1 = sum(-(maxLambda ** (n - i)) * p[i] for i in range(n + 1))
print("\nНевязка Pn(lambda^k):\n\t", r1)