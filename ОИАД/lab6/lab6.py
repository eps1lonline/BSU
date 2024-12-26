import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. Генерация точек для двух классов
np.random.seed(0)

# Класс 1 (p1)
p1 = np.random.randn(100, 2) + [2, 2]  # смещаем группу в точку (2, 2)

# Класс 2 (p2)
p2 = np.random.randn(100, 2) + [4, 4]  # смещаем группу в точку (4, 4)

# Визуализация точек
plt.scatter(p1[:, 0], p1[:, 1], color='red', label='Класс 1')
plt.scatter(p2[:, 0], p2[:, 1], color='blue', label='Класс 2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Группы точек p1 и p2')
plt.show()

# 2. Разделение на тренировочную и тестовую выборки
labels_p1 = np.zeros(len(p1))  # Класс 1
labels_p2 = np.ones(len(p2))   # Класс 2

# Объединение данных
X = np.vstack([p1, p2])
y = np.hstack([labels_p1, labels_p2])

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Визуализация тренировочных данных
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Класс 1 (train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Класс 2 (train)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Тренировочные данные')
plt.show()

# 3. Создание классификаторов
nb = GaussianNB()
svm = SVC(probability=True)
log_reg = LogisticRegression()
tree = DecisionTreeClassifier()

# Усреднение классификаторов
voting_clf = VotingClassifier(estimators=[
    ('naive_bayes', nb),
    ('svm', svm),
    ('log_reg', log_reg),
    ('tree', tree)
], voting='soft')

# Список всех классификаторов
classifiers = [nb, svm, log_reg, tree, voting_clf]

# Тренировка классификаторов
for clf in classifiers:
    clf.fit(X_train, y_train)

# 4. Оценка классификации и построение таблицы ошибок
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, clf in enumerate(classifiers):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                xticklabels=['Класс 1', 'Класс 2'], yticklabels=['Класс 1', 'Класс 2'])
    axes[i].set_title(f'{clf.__class__.__name__}')
    
plt.tight_layout()
plt.show()

# 5. Генерация группы точек p3 и классификация
p3 = np.random.randn(50, 2) + [3, 3]  # Группа пересекается с p1 и p2

# Классификация точек p3 всеми классификаторами
p3_predictions = {}
for clf in classifiers:
    p3_predictions[clf.__class__.__name__] = clf.predict(p3)
    print(f'{clf.__class__.__name__} \tsum={np.sum(p3_predictions[clf.__class__.__name__])}')

# Визуализация результатов
plt.scatter(p1[:, 0], p1[:, 1], color='red', label='Класс 1')
plt.scatter(p2[:, 0], p2[:, 1], color='blue', label='Класс 2')
plt.scatter(p3[:, 0], p3[:, 1], color='green', label='p3', marker='x')

# Отображение меток для p3
for clf_name, predictions in p3_predictions.items():
    print(f"{clf_name} predictions for p3: {predictions}")

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Классификация точек p3')
plt.show()
