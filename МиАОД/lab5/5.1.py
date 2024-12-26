import pandas as pd
from scipy import stats

# Загрузка данных
data = pd.read_csv("C:/Users/nikit/Desktop/creditcard.csv")

# Вычисление Z-score
z_scores = stats.zscore(data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 
                                'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 
                                'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 
                                'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
abs_z_scores = abs(z_scores)
threshold = 3  # Пороговое значение для Z-score

# Определение аномалий
anomalies_z = (abs_z_scores > threshold).any(axis=1)
data['Anomaly_Z'] = anomalies_z

# Вывод аномалий
print(data[data['Anomaly_Z']])