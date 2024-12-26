
import numpy as np
import matplotlib.pyplot as plt

# Параметры
r = 0.05  # Базовая ставка дисконтирования (5%)
n_years = 30  # Горизонт расчетов

# Платежные потоки
cashflows_annuity = [0] * 14 + [1000] * (n_years - 14)  # Аннуитет Поток платежей в размере 1000 за 16 лет
cashflows_life = [0, 0, 0, 0, 10000] + [0] * (n_years - 5)  # Страхование дожития Платеж в 10,000 через 5 лет

# print(f'{cashflows_annuity} \n{cashflows_life}')

# Суммарный поток пассивов
cashflows_liabilities = [a + l for a, l in zip(cashflows_annuity, cashflows_life)]

print(cashflows_liabilities)
# избыточные ресурсы

# Функция для расчета PV, дюрации и выпуклости с учетом дисконтирования
def calculate_metrics_with_discounting(cashflows, r):
    k = np.arange(1, len(cashflows) + 1)  # Периоды
    discount_factors = np.exp(-r * k)  # Дисконтирующие множители
    discounted_cashflows = np.array(cashflows) * discount_factors  # Дисконтированные потоки
    
    pv = np.sum(discounted_cashflows)  # Текущая стоимость
    duration = np.sum(k * discounted_cashflows) / pv  # Дюрация
    convexity = np.sum(k**2 * discounted_cashflows) / pv  # Выпуклость
    return pv, duration, convexity

# Расчет PV, дюрации и выпуклости для пассивов
pv_liabilities, duration_liabilities, convexity_liabilities = calculate_metrics_with_discounting(cashflows_liabilities, r)

# Облигации
bond_cashflows = {
    "5y": [8] * 4 + [108] + [0] * (n_years - 5),  # 8% купон, погашение через 5 лет
    "15y": [8] * 14 + [108] + [0] * (n_years - 15),  # 8% купон, погашение через 15 лет
    "30y": [0] * 29 + [100],  # Без купона, погашение через 30 лет
}

# print(f'{bond_cashflows}')

# Расчет для облигаций с учетом дисконтирования
bond_metrics = {}
for bond, cashflows in bond_cashflows.items():
    bond_metrics[bond] = calculate_metrics_with_discounting(cashflows, r)

# Составляем матрицу A и вектор b для иммунизации
A = np.array([
    [1, 1, 1],  # Сумма долей должна быть 1
    [bond_metrics["5y"][1], bond_metrics["15y"][1], bond_metrics["30y"][1]],  # Дюрации
    [bond_metrics["5y"][2], bond_metrics["15y"][2], bond_metrics["30y"][2]],  # Выпуклости
])
b = np.array([1, duration_liabilities, convexity_liabilities])  # Нормализуем PV = 1

print(f'A={A} \nb={b}')

# Решение системы уравнений
x = np.linalg.solve(A, b)
x_5y, x_15y, x_30y = x  # Доли активов в 5-, 15-, и 30-летних облигациях

# Проверка PV, Duration, Convexity для активов
pv_assets = np.dot(x, [bond_metrics["5y"][0], bond_metrics["15y"][0], bond_metrics["30y"][0]])
duration_assets = np.dot(x, [bond_metrics["5y"][1], bond_metrics["15y"][1], bond_metrics["30y"][1]])
convexity_assets = np.dot(x, [bond_metrics["5y"][2], bond_metrics["15y"][2], bond_metrics["30y"][2]])

# Сценарии изменения процентной ставки (New York Seven)
delta_rates = np.linspace(-0.02, 0.02, 7)  # Изменения ставки от -2% до +2%

# Функция для оценки изменений PV (сюрплюс)
def calculate_surplus(delta_r, pv_assets, duration_assets, convexity_assets):
    pv_change_assets = -duration_assets * delta_r * pv_assets + 0.5 * convexity_assets * (delta_r ** 2) * pv_assets
    pv_change_liabilities = -duration_liabilities * delta_r * pv_liabilities + 0.5 * convexity_liabilities * (delta_r ** 2) * pv_liabilities
    return pv_change_assets - pv_change_liabilities

# Расчет сюрплюса
surpluses = [calculate_surplus(delta_r, pv_assets, duration_assets, convexity_assets) for delta_r in delta_rates]



import numpy as np

# Параметры
n_years = 30  # Горизонт расчетов

# Платежи для облигаций
bond_cashflows = {
    "5y": [8] * 4 + [108] + [0] * (n_years - 5),  # 8% купон, погашение через 5 лет
    "15y": [8] * 14 + [108] + [0] * (n_years - 15),  # 8% купон, погашение через 15 лет
    "30y": [0] * 29 + [100],  # Без купона, погашение через 30 лет
}

# Вывод платежей для каждой облигации
for bond, cashflows in bond_cashflows.items():
    print(f"{bond} платежи на: {cashflows[:30]}")

# Вывод результатов
print("Доли активов:")
print(f"5-летние облигации: {x_5y:.2%}")
print(f"15-летние облигации: {x_15y:.2%}")
print(f"30-летние облигации: {x_30y:.2%}")

# График сюрплюса
plt.figure(figsize=(10, 6))
plt.plot(delta_rates, surpluses, marker='o', linestyle='-', color='b')
plt.axhline(0, color='r', linestyle='--', linewidth=1)
plt.title("Сюрплюс при изменении процентной ставки")
plt.xlabel("Изменение процентной ставки (Δr)")
plt.ylabel("Сюрплюс")
plt.grid(True)
plt.show()
