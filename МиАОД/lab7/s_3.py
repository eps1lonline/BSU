# Задание 3: Использование метода-обертки при кросс-валидации.
# Используйте любой набор данных на ваше усмотрение. 
# Выберите модель машинного обучения и метод-обертку для отбора признаков. 
# Примените кросс-валидацию, чтобы оценить эффективность этого подхода.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import numpy as np

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# Определение модели
model = LogisticRegression(max_iter=200)

# Определение RFE с Logistic Regression
rfe = RFE(estimator=model, n_features_to_select=2)

# Создание пайплайна
pipeline = Pipeline(steps=[('feature_selection', rfe), ('classification', model)])

# Определение кросс-валидации
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Оценка модели с использованием кросс-валидации
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')

# Вывод результатов
print("Accuracy scores for each fold: ", scores)
print("Mean accuracy: ", np.mean(scores))