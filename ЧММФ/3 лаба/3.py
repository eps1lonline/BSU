import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_grids(left_border, right_border, num_x_points, upper_bound, num_t_points):
    h = (right_border-left_border) / num_x_points
    nodes_x = np.linspace(left_border, right_border, num_x_points+1)

    tau = upper_bound / num_t_points
    nodes_t = np.linspace(0, upper_bound, num_t_points+1)

    print('h =', h)
    print('tau =', tau)

    return nodes_x, nodes_t, h, tau

def u(x, t, a, u_0):
    return u_0(x-a*t)

a = 1
def u_0(x):
    return x**2

def diff_scheme_solve(nodes_x, nodes_t, h, tau, u_0, a):
    gamma = a * tau / h
    
    y = np.zeros((len(nodes_x), len(nodes_t)))
    for k in range(len(nodes_x)):
        y[k, 0] = u_0(nodes_x[k])

    for k in range(len(nodes_x)-1):
        for j in range(len(nodes_t)-1):
            y[k, j+1] = (1-gamma) * y[k, j] + gamma * y[k+1, j]
    return y

nodes_x, nodes_t, h, tau = generate_grids(0, 1, 5, 0.25, 5)

y = diff_scheme_solve(nodes_x, nodes_t, h, tau, u_0, a)

plt.figure(figsize=(16, 8))  # Создаем одну область для всех графиков

for j, t in enumerate(nodes_t):
    plt.plot(nodes_x[:-1], y[:-1, j], label=f'numerical solution (t={round(t, 2)})')
    plt.plot(nodes_x, u(nodes_x, t, a, u_0), label=f'exact solution (t={round(t, 2)})')

plt.grid(True)
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Аппроксимация для всех временных шагов')
plt.legend()
plt.show()