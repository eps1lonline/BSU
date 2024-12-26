import pandas as pd

data = pd.read_csv("C:/Users/nikit/Desktop/covid19_tweets.csv") # Загрузка данных из файла

actualDate = '2020-08-10'
data['date'] = pd.to_datetime(data['date']) # Преобразование столбца 'dt' в формат datetime
filtered_data = data[data['date'] > actualDate] # Фильтрация данных
result = filtered_data[['user_name', 'date']] # Выбор только нужных столбцов
print(result)   