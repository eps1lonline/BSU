import pandas as pd

# Создаем датафрейм
df = pd.DataFrame({
    'уровень образования': ['среднее', 'высшее', 'начальное'],
    'согласие': ['да', 'нет', 'да']
})

# Функция для преобразования в порядковую шкалу
def convert_to_ordinal(data, categories):
    return data.astype(pd.CategoricalDtype(categories=categories, ordered=True))

# Преобразование уровня образования в порядковую шкалу
df['уровень образования'] = convert_to_ordinal(df['уровень образования'], ['начальное', 'среднее', 'высшее'])

# Преобразование согласия в номинальную шкалу (категориальные данные)
df['согласие'] = df['согласие'].astype('category')

print(df)