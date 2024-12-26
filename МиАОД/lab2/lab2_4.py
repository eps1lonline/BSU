import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Создаем датафрейм с различными типами данных: номинальные, порядковые, интервальные и отношения
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female'],  # Номинальная шкала
    'Age': [23, 31, 28, 22, 35, 30],  # Интервальная шкала
    'Education_Level': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],  # Порядковая шкала
    'Test_Score': [88, 95, 75, 85, 90, 92],  # Шкала отношений
}

df = pd.DataFrame(data)

# Определяем порядок для порядковой шкалы
education_order = ['High School', 'Bachelor', 'Master', 'PhD']

# Столбчатая диаграмма для номинальной шкалы (Пол)
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution (Nominal Scale)')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Столбчатая диаграмма для порядковой шкалы (Уровень образования)
plt.figure(figsize=(6, 4))
sns.countplot(x='Education_Level', data=df, order=education_order)
plt.title('Education Level Distribution (Ordinal Scale)')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

# Гистограмма для интервальной шкалы (Возраст)
plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=5, kde=True)
plt.title('Age Distribution (Interval Scale)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Гистограмма для шкалы отношений (Баллы за тест)
plt.figure(figsize=(6, 4))
sns.histplot(df['Test_Score'], bins=5, kde=True)
plt.title('Test Score Distribution (Ratio Scale)')
plt.xlabel('Test Score')
plt.ylabel('Frequency')
plt.show()
