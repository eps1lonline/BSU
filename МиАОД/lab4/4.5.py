import pandas as pd
import folium

# Загрузка данных из CSV файла с указанием кодировки
data_path = "C:/Users/nikit/Desktop/globalterrorismdb_0718dist.csv"
gtd_data = pd.read_csv(data_path, encoding='ISO-8859-1')  # Попробуйте 'Windows-1251', если это не сработает
# print(gtd_data.head())

# Фильтрация данных для использования только тех записей, где есть координаты
gtd_data = gtd_data[gtd_data['latitude'].notnull() & gtd_data['longitude'].notnull()]

# Создание карты с начальной точкой
map_center = [gtd_data['latitude'].mean(), gtd_data['longitude'].mean()]
gtd_map = folium.Map(location=map_center, zoom_start=2)

# Добавление точек на карту
for _, row in gtd_data.iterrows():
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=5,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        popup=f"Событие ID: {row['eventid']}<br>Страна: {row['country_txt']}<br>Город: {row['city']}"
    ).add_to(gtd_map)

# Сохранение карты в HTML файл
gtd_map.save("C:/Users/nikit/Desktop/gtd_map.html")
print("end")