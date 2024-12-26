import pandas as pd
import numpy as np

# Создаем датафрейм с различными типами данных: номинальные, порядковые, интервальные и отношения
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Male'],  # Номинальная шкала
    'Age': [23, 31, 28, 22, 35, 30],  # Интервальная шкала
    'Education_Level': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],  # Порядковая шкала
    'Test_Score': [88, 95, 75, 85, 90, 92],  # Шкала отношений
}

df = pd.DataFrame(data)

# Анализ данных
# Номинальная шкала: мода
gender_mode = df['Gender'].mode()[0]

# Порядковая шкала: медиана
# Для порядка преобразуем уровни образования в числовой формат
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
df['Education_Level_Ordered'] = df['Education_Level'].apply(lambda x: education_order.index(x))
education_median = np.median(df['Education_Level_Ordered'])
education_median_level = education_order[int(education_median)]

# Интервальная шкала: среднее значение
age_mean = np.mean(df['Age'])

# Шкала отношений: среднее значение
test_score_mean = np.mean(df['Test_Score'])

# Результаты анализа
print({
    'Gender Mode': gender_mode,
    'Education Level Median': education_median_level,
    'Age Mean': age_mean,
    'Test Score Mean': test_score_mean
})
