import numpy as np
import scipy
from scipy import integrate
import math

print('Метод наименьших квадратов')
def f(x):
 return 0.2 * np.exp(x) + 0.8 * np.cos(x)
def s(i, j):
 return 1. / (i + j + 1)
def m(i):
 return integrate.quad(lambda x: pow(x, i) * (0.2 * np.exp(x) + 0.8 * np.cos(x)), 0, 1)
def poly_val(coefs, x):
 i = 0
 res = 0.0
 for coef in coefs:
  res += coef * pow(x,i)
  i += 1
 return res

X = [0.27, 0.7, 1.17]
A = np.array([[s(i, j) for i in range(6)] for j in range(6)])
b = np.array([m(i)[0] for i in range(6)])

coefs = np.linalg.solve(A, b)
for x in X:
 print('P(', x, ') = ', poly_val(coefs, x), sep='')
 print('f(', x, ') = ', f(x), sep='')
 print('|P(', x, ') - f(',x,')| = ', abs( poly_val(coefs, x) - f(x)))
 print()


print('\n\nЛагранж')
def w(x, nodes):
 res = 1.0
 for node in nodes:
  res *= (x - node)
 return res
def lagrange_error(M, x, nodes):
 return np.abs(M * w(x, nodes) / math.factorial(nodes.size))
def lagrange_value(x, nodes):
 res = 0.0
 for i in range(nodes.size):
  p = 1.0
  for j in range(nodes.size):
   if i == j:
    continue
   p *= (x - nodes[j])
   p /= (nodes[i] - nodes[j])
  res += p * f(nodes[i])
 return res
interpolation_nodes = np.linspace(0.0, 1.0, num=11)
for x in X:
 print('P(', x, ') = ', lagrange_value(x,interpolation_nodes), sep='')
 print('f(', x, ') = ', f(x), sep='')
 print('r(', x, ') = ', lagrange_error(1.21,x,interpolation_nodes), sep='')
 print('|P(', x, ') - f(', x, ')| = ', abs(lagrange_value(x,interpolation_nodes) - f(x)))
 print()


print('\n\nНьютон')
def build_table(nodes): # построение таблицы разделенных разностей
 table = np.zeros(shape=(nodes.size, nodes.size))
 for i in range(nodes.size):
  table[i, 0] = f(nodes[i])
 for i in range(1, nodes.size):
  for j in range(nodes.size - i):
   table[j, i] = (table[j + 1, i - 1] - table[j, i - 1]) / (i * 0.1)
 return table
def newton_value(x, rr_table, nodes):
 n = rr_table.shape[0]
 p = 1.0
 res = rr_table[0, 0]
 for i in range(1, n):
  p *= (x - nodes[i - 1])
  res += p * rr_table[0, i]
 return res

rr_table = build_table(interpolation_nodes)
for x in X:
 print('P(', x, ') = ', newton_value(x,rr_table, interpolation_nodes), sep='')
 print('f(', x, ') = ', f(x), sep='')
 print('|P(', x, ') - f(', x, ')| = ', abs(newton_value(x,rr_table, interpolation_nodes) - f(x)))
 print()


 def chebyshev_nodes(n, a, b):
  res = []
  for i in range(n):
   x = (a + b) / 2 + (b - a) / 2 * np.cos(np.pi * (2 * i + 1) / (2 * n))
   res.append(x)
  return np.array(res)


 print('\n\nУзлы выбранные наилучшим образом: ')
 best_nodes = chebyshev_nodes(11, 0, 1)
 print(best_nodes)
 for x in X:
  print('P(', x, ') = ', lagrange_value(x, best_nodes), sep='')
  print('f(', x, ') = ', f(x), sep='')
  print('r(', x, ') = ', lagrange_error(0.3, x, best_nodes), sep='')
  print('|P(', x, ') - f(', x, ')| = ', abs(lagrange_value(x, best_nodes) - f(x)))
  print()