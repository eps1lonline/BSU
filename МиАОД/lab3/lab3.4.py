import pandas as pd
import pycountry

# Создание искусственного набора данных о заболеваемости раком
data = {
    'Country': ['USA', 'Canada', 'Germany', 'France', 'Italy', 'Spain', 'Brazil', 'Argentina', 'Australia', 'India'],
    'CancerType': ['Lung', 'Breast', 'Prostate', 'Lung', 'Colon', 'Liver', 'Breast', 'Prostate', 'Lung', 'Colon'],
    'Incidence': [200000, 150000, 80000, 60000, 90000, 30000, 50000, None, 40000, 70000],  # Наличие пропущенного значения
    'Mortality': [150000, 100000, 40000, 30000, 50000, 20000, 25000, 10000, 15000, 30000]
}

# Преобразование в DataFrame
df = pd.DataFrame(data)

# Обзор данных
print("Первые несколько строк данных:")
print(df)

# Проверка уникальных стран
unique_countries = df['Country'].unique()
print("\nУникальные страны в наборе данных:")
print(unique_countries)

# Сравнение с полным списком стран
all_countries = [country.name for country in pycountry.countries]
missing_countries = set(all_countries) - set(unique_countries)

print("\nОтсутствующие страны:")
print(missing_countries)

# Анализ пропущенных значений
missing_values = df.isnull().sum()
print("\nКоличество пропущенных значений в каждом столбце:")
print(missing_values)

# Вывод информации о пропущенных значениях
if missing_values.any():
    print("\nНекоторые столбцы имеют пропущенные значения.")
else:
    print("\nНет пропущенных значений в столбцах.")