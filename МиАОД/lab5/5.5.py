import seaborn as sns

titanic = sns.load_dataset('titanic')

print(titanic.isnull().sum())

titanic_dropped = titanic.dropna()
print(f"Количество строк после удаления пропущенных значений: {len(titanic_dropped)}")

titanic_mean = titanic.copy()
for column in titanic_mean.select_dtypes(include='number').columns:
    titanic_mean[column] = titanic_mean[column].fillna(titanic_mean[column].mean())

print(f"Количество строк после заполнения средним значением: {len(titanic_mean)}")

print("Результаты после удаления пропущенных значений:")
print(titanic_dropped.isnull().sum(),"\n")

print("Результаты после заполнения средним значением:")
print(titanic_mean.isnull().sum(),"\n")