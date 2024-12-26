import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузка данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Разделим данные на признаки и целевую переменную
X = data.drop('quality', axis=1)
y = data['quality']

# Шаг 2: Корреляция Спирмена
spearman_corr = data.corr(method='spearman')

# Визуализация корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Корреляция Спирмена между признаками')
plt.show()

# Отбор признаков по корреляции
correlation_threshold = 0.3  # Порог для отбора
correlated_features = spearman_corr.index[abs(spearman_corr['quality']) > correlation_threshold].tolist()
print("Признаки отобранные по корреляции Спирмена:", correlated_features)

# Шаг 3: Метод отбора на основе дерева решений
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Важность признаков
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Важность признаков (дерево решений):")
for f in range(X.shape[1]):
    print(f"{X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

# Шаг 4: Метод Lasso
lasso = LassoCV(cv=5)
lasso.fit(X_train, y_train)

# Признаки, выбранные Lasso
lasso_selected_features = X.columns[np.abs(lasso.coef_) > 1e-4]
print("Признаки отобранные методом Lasso:", lasso_selected_features.tolist())

# Шаг 5: Сравнение результатов
print("\nСравнение методов отбора признаков:")
print("1. Корреляция Спирмена:", correlated_features)
print("2. Дерево решений:", [X.columns[i] for i in indices if importances[i] > 0])
print("3. Lasso:", lasso_selected_features.tolist())