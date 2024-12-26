import pandas as pd
import plotly.express as px

# Пример данных о выборах с сокращениями для штатов
data = {
    'State': ['CA', 'TX', 'FL', 'NY', 'IL'],  # Сокращения для штатов
    'Votes_Democrat': [1000000, 500000, 700000, 900000, 600000],
    'Votes_Republican': [400000, 3000000, 2000000, 300000, 1500000]
}

# Создание DataFrame
df = pd.DataFrame(data)

# Создание интерактивной карты
fig_map = px.choropleth(
    df,
    locations='State',
    locationmode='USA-states',
    color='Votes_Democrat',
    scope='usa',
    title='Голосование за Демократов по штатам',
    color_continuous_scale='Blues',
    labels={'Votes_Democrat': 'Голоса за Демократов'}
)

# Отображение карты
fig_map.show()

# Создание интерактивной диаграммы с подсказками
fig_bar = px.bar(
    df,
    x='State',
    y=['Votes_Democrat', 'Votes_Republican'],
    title='Голоса по партиям в 2020 году',
    labels={'value': 'Количество голосов', 'variable': 'Партия'},
    barmode='group'
)

# Отображение диаграммы
fig_bar.show()