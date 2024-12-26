import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 1.0
nx = 10
nt = 1000
dx = L / (nx - 1)
dt = T / (nt - 1)

alpha = dt / dx**2
print(alpha)

if alpha > 0.5:
    print("Условие устойчивости нарушено! Уменьшите dt или увеличьте nx.")
    exit()

x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

u_num = np.zeros((nt, nx))
u_exact = np.zeros((nt, nx))

u_num[0, :] = np.cos(x)  
u_exact[0, :] = np.cos(x) 

u_num[:, -1] = np.cos(1 + t) 

for n in range(0, nt - 1):
    for i in range(1, nx - 1):
        u_num[n + 1, i] = (
            u_num[n, i]
            + alpha * (u_num[n, i + 1] - 2 * u_num[n, i] + u_num[n, i - 1])
            + dt * np.sqrt(2) * np.sin(np.pi / 4 * x[i] - t[n])
        )
    u_num[n + 1, 0] = (
        u_num[n + 1, 1]
        - dx * (u_num[n + 1, 0] - np.sqrt(2) * np.sin(np.pi / 4 + t[n + 1]))
    )

for n in range(nt):
    for i in range(nx):
        u_exact[n, i] = np.cos(x[i] + t[n])

plt.figure(figsize=(10, 6))
for n in range(0, nt, nt // 5):  
    plt.plot(x, u_num[n, :], label=f"Численное решение, t = {t[n]:.2f}")
    plt.plot(x, u_exact[n, :], "--", label=f"Точное решение, t = {t[n]:.2f}")

plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Сравнение численного и точного решений")
plt.legend()
plt.grid()
plt.show()