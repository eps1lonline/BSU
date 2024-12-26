import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom, poisson, chi2, norm, expon, binom, poisson

# 1.1
file_path = 'C:/Users/nikit/Desktop/avianHabitat.csv'
data = pd.read_csv(file_path)
selected_columns = ['VOR', 'PDB', 'DBHt', 'PW', 'WHt', 'PE', 'EHt', 'PA', 'AHt', 'PH', 'HHt', 'PL', 'LHt', 'PB']
filtered_data = data[selected_columns].copy()
print(filtered_data)

def remove_outliers(column, num, selected_columns):
    std_dev = column.std()  # стандартное отклонение
    print(num, "\t", selected_columns[num - 1], "\tstd =", std_dev)
    mean = column.mean()
    threshold = mean + 2 * std_dev
    return column[column <= threshold]

for num, column in enumerate(filtered_data.columns):
    filtered_data[column] = remove_outliers(filtered_data[column], num, selected_columns)
print(filtered_data)

# 1.2
def discretize_column(column, num_bins):
    return pd.cut(column, bins=num_bins, labels=False)

num_bins = 8
for column in filtered_data.columns:
    filtered_data[column] = discretize_column(filtered_data[column].dropna(), num_bins)
print(filtered_data)

# 2, 3.1
# нормальное распределение
cleaned_data = filtered_data['VOR'].dropna() 
mean = cleaned_data.mean()
std_dev = cleaned_data.std()
x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100) # Параметры нормального распределения
normal_distribution = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

plt.figure(figsize=(10, 6))
sns.histplot(cleaned_data, bins=10, kde=False, color='blue', stat='density', label='Выборочные данные')
plt.plot(x, normal_distribution, color='red', label='Нормальное распределение', linewidth=2)
plt.title('Сравнение выборки и нормального распределения')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid()
plt.show()

# экспоненциальное распределение
lambda_param = 1 / cleaned_data.mean() # Параметр λ
x = np.linspace(0, cleaned_data.max(), 100) # Генерация значений для экспоненциального распределения
exponential_distribution = lambda_param * np.exp(-lambda_param * x)

plt.figure(figsize=(10, 6))
sns.histplot(cleaned_data, bins=10, kde=False, color='blue', stat='density', label='Выборочные данные')
plt.plot(x, exponential_distribution, color='red', label='Экспоненциальное распределение', linewidth=2)
plt.title('Сравнение выборки и экспоненциального распределения')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid()
plt.show()

# биномиальное распределение
n = int(cleaned_data.max()) # Количество испытаний
p = cleaned_data.mean() / n # Вероятность успеха
x = np.arange(0, n + 1) # Генерация значений для биномиального распределения
binomial_distribution = binom.pmf(x, n, p)

plt.figure(figsize=(10, 6))
sns.histplot(cleaned_data, bins=n, kde=False, color='blue', stat='density', label='Выборочные данные', discrete=True)
plt.stem(x, binomial_distribution, linefmt='red', markerfmt='ro', basefmt=' ', label='Биномиальное распределение')
plt.title('Сравнение выборки и биномиального распределения')
plt.xlabel('Количество успехов')    
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid()
plt.show()

# пуассоновское распределение
lambda_hat = np.mean(cleaned_data)
mu_hat = np.mean(cleaned_data)
sigma_hat = np.std(cleaned_data)

plt.figure(figsize=(10, 6))
x_poisson = np.arange(0, int(max(cleaned_data)) + 1)
poisson_probs = poisson.pmf(x_poisson, lambda_hat)
plt.bar(x_poisson, poisson_probs, color='blue', alpha=0.6, label='Пуассон')
plt.title('Пуассоновское распределение')
plt.xlabel('Число событий')
plt.ylabel('Вероятность')
plt.xticks(x_poisson)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 3.2
print("Оценки параметров распределений:")
print(f"Нормальное распределение: μ = {mean:.2f}, σ = {std_dev:.2f}")
print(f"Экспоненциальное распределение: λ = {lambda_param:.2f}")
print(f"Биномиальное распределение: n = {n}, p = {p:.2f}")
print(f"Пуассоновское распределение: λ = {lambda_hat:.2f}")
print()

# 4
def chi_squared_test(observed, expected, ddof):
    chi_squared_stat = np.sum(np.where(expected != 0, (observed - expected) ** 2 / expected, 0))
    p_value = 1 - chi2.cdf(chi_squared_stat, df=ddof)
    return chi_squared_stat, p_value

observed_freq, bin_edges = np.histogram(cleaned_data, bins=10, density=False)# Наблюдаемые частоты

normal_expected_freq = len(cleaned_data) * ( # Ожидаемые частоты для нормального распределения
    norm.cdf(bin_edges[1:], mean, std_dev) -
    norm.cdf(bin_edges[:-1], mean, std_dev)
)

exponential_expected_freq = len(cleaned_data) * ( # Ожидаемые частоты для экспоненциального распределения
    expon.cdf(bin_edges[1:], scale=1/lambda_param) -
    expon.cdf(bin_edges[:-1], scale=1/lambda_param)
)

binomial_expected_freq = len(cleaned_data) * binom.pmf(np.arange(0, len(bin_edges)-1), n, p) # Ожидаемые частоты для биномиального распределения
poisson_expected_freq = len(cleaned_data) * poisson.pmf(np.arange(0, len(bin_edges)-1), lambda_hat) # Ожидаемые частоты для пуассоновского распределения

# Проведение теста хи-квадрат
ddof = len(bin_edges) - 2 # Скорректируйте число степеней свободы при необходимости

chi2_normal, p_normal = chi_squared_test(observed_freq, normal_expected_freq, ddof)
chi2_exponential, p_exponential = chi_squared_test(observed_freq, exponential_expected_freq, ddof)
chi2_binomial, p_binomial = chi_squared_test(observed_freq, binomial_expected_freq, ddof)
chi2_poisson, p_poisson = chi_squared_test(observed_freq, poisson_expected_freq, ddof)

print("Результаты теста хи-квадрат:")
print(f"Нормальное распределение: χ² = {chi2_normal:.2f}, p = {p_normal:.2f}")
print(f"Экспоненциальное распределение: χ² = {chi2_exponential:.2f}, p = {p_exponential:.2f}")
print(f"Биномиальное распределение: χ² = {chi2_binomial:.2f}, p = {p_binomial:.2f}")
print(f"Пуассоновское распределение: χ² = {chi2_poisson:.2f}, p = {p_poisson:.2f}")
print()

# 5
best_fit = max([(chi2_normal, p_normal, 'Нормальное распределение'),
                (chi2_exponential, p_exponential, 'Экспоненциальное распределение'),
                (chi2_binomial, p_binomial, 'Биномиальное распределение'),
                (chi2_poisson, p_poisson, 'Пуассоновское распределение')],
               key=lambda x: x[1])

print(f"Лучшее приближение: {best_fit[2]} с p-значением {best_fit[1]:.2f}")