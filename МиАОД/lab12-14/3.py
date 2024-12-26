# Импорт необходимых библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Загрузка данных из CSV файла
# Замените 'wdbc.data' на путь к вашему файлу
data = pd.read_csv('wdbc.data', header=None)

# Название колонок
data.columns = [
    'ID', 'Diagnosis',
    'Radius_Mean', 'Texture_Mean', 'Perimeter_Mean', 'Area_Mean',
    'Smoothness_Mean', 'Compactness_Mean', 'Concavity_Mean', 
    'Concave_Points_Mean', 'Symmetry_Mean', 'Fractal_Dimension_Mean',
    'Radius_SE', 'Texture_SE', 'Perimeter_SE', 'Area_SE',
    'Smoothness_SE', 'Compactness_SE', 'Concavity_SE', 
    'Concave_Points_SE', 'Symmetry_SE', 'Fractal_Dimension_SE',
    'Radius_Worst', 'Texture_Worst', 'Perimeter_Worst', 
    'Area_Worst', 'Smoothness_Worst', 'Compactness_Worst', 
    'Concavity_Worst', 'Concave_Points_Worst', 'Symmetry_Worst', 
    'Fractal_Dimension_Worst'
]

# Преобразование целевой переменной в числовой формат
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Разделение данных на признаки и целевую переменную
X = data.drop(['ID', 'Diagnosis'], axis=1)
y = data['Diagnosis']

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Создание и обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка модели
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# Визуализация матрицы ошибок
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанная метка')
plt.ylabel('Истинная метка')
plt.title('Матрица ошибок')
plt.show()