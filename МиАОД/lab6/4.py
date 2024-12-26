# измеряет степень зависимости между двумя переменными. помогает определить, насколько хорошо один признак может предсказать другую переменную.
# Импорт необходимых библиотек
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка набора данных о цветах ирисов
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Применение метода взаимной информации для отбора признаков
mi = mutual_info_classif(X, y, random_state=42)

# Создание DataFrame для удобного отображения результатов
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})

# Сортировка по взаимной информации
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Вывод результатов
print("Взаимная информация признаков:")
print(mi_df)

# Визуализация результатов
plt.figure(figsize=(10, 6))
sns.barplot(x='Mutual Information', y='Feature', data=mi_df, palette='viridis')
plt.title('Отбор признаков с помощью взаимной информации')
plt.xlabel('Взаимная информация')
plt.ylabel('Признаки')
plt.show()