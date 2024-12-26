import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from sklearn.datasets import make_moons


def task1_1():
    penguins = sns.load_dataset("penguins")
    print(penguins.head())

    print(penguins.isnull().sum())

    imputer = SimpleImputer(strategy="most_frequent")
    penguins_cleaned = pd.DataFrame(imputer.fit_transform(penguins), columns=penguins.columns)

    # Выделение числовых признаков
    numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    data = penguins_cleaned[numeric_features].astype(float)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    print(pd.DataFrame(data_scaled, columns=numeric_features).head())

    # K-Means
    # Метод локтя
    inertia = []
    range_n_clusters = range(1, 10)

    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    # График метода локтя
    plt.plot(range_n_clusters, inertia, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal Clusters")
    plt.show()

    # K-Means с 3 кластерами
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters_kmeans = kmeans.fit_predict(data_scaled)
    penguins_cleaned["Cluster_KMeans"] = clusters_kmeans

    linkage_matrix = linkage(data_scaled, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix)
    plt.title("Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

    # Агломеративная кластеризация
    agg_clustering = AgglomerativeClustering(n_clusters=3)
    clusters_agg = agg_clustering.fit_predict(data_scaled)
    penguins_cleaned["Cluster_Agglomerative"] = clusters_agg

    # DBSCAN кластеризация
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters_dbscan = dbscan.fit_predict(data_scaled)
    penguins_cleaned["Cluster_DBSCAN"] = clusters_dbscan

    # Визуализация кластеров с использованием PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Визуализация кластеров для K-Means
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_kmeans, cmap='viridis')
    plt.title("K-Means Clustering")
    plt.show()

    # Визуализация кластеров для агломеративной кластеризации
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_agg, cmap='viridis')
    plt.title("Agglomerative Clustering")
    plt.show()

    # Визуализация кластеров для DBSCAN
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_dbscan, cmap='viridis')
    plt.title("DBSCAN Clustering")
    plt.show()

    # Сравнение результатов кластеризации
    comparison_kmeans = penguins_cleaned.groupby("Cluster_KMeans")[numeric_features].mean()
    print("Средние значения признаков по кластерам (K-Means):")
    print(comparison_kmeans)

    comparison_agg = penguins_cleaned.groupby("Cluster_Agglomerative")[numeric_features].mean()
    print("Средние значения признаков по кластерам (Agglomerative):")
    print(comparison_agg)

    unique_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
    print(f"Количество кластеров, найденных DBSCAN: {unique_clusters_dbscan}")

task1_1()



def task1_2():
    
    data = pd.read_csv("D:\\Desktop\\7 sem\\МиАОД\\lab15-17\\uci-news-aggregator.csv")

    headlines = data["TITLE"]
    print(f"Всего заголовков: {len(headlines)}")

    # Предобработка текста
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", text) 
        text = text.lower()
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Лемматизация и удаление стоп-слов
        return " ".join(words)

    headlines_cleaned = headlines.apply(preprocess_text)
    print(headlines_cleaned.head())

    # Преобразование текста в числовой вид
    # Векторизация с помощью TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Оставляем топ-1000 признаков
    tfidf_matrix = tfidf_vectorizer.fit_transform(headlines_cleaned)

    # Преобразованный массив
    print(tfidf_matrix.shape)  # Размерность: (число документов, 1000 признаков)

    # K-Means
    # Метод локтя
    inertia = []
    range_n_clusters = range(1, 10)

    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tfidf_matrix)
        inertia.append(kmeans.inertia_)

    # График для метода локтя
    plt.plot(range_n_clusters, inertia, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal Clusters")
    plt.show()

    # K-Means с 5 кластерами
    optimal_clusters = 5
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # Добавление результатов в DataFrame
    data["Cluster"] = clusters
    print(data.groupby("Cluster")["TITLE"].count())

    # Анализ кластеров
    # Вывод ключевых слов для каждого кластера
    terms = tfidf_vectorizer.get_feature_names_out()
    centroids = kmeans.cluster_centers_

    for i in range(optimal_clusters):
        cluster_terms = centroids[i].argsort()[-10:][::-1]  # Топ-10 ключевых слов
        print(f"Кластер {i}: {', '.join(terms[t] for t in cluster_terms)}")

    # Примеры заголовков из каждого кластера
    for i in range(optimal_clusters):
        print(f"\nКластер {i}:")
        print(data[data["Cluster"] == i]["TITLE"].head(5).values)

    # Визуализация кластеров
    # PCA для уменьшения размерности
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(tfidf_matrix.toarray())

    # Визуализация кластеров
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=5)
    plt.title("Кластеры новостных заголовков")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

#task1_2()



def task1_3():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    # Определение оптимального числа кластеров
    # Метод локтя
    inertia = []
    range_n_clusters = range(1, 10)

    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    # График метода локтя
    plt.plot(range_n_clusters, inertia, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal Clusters")
    plt.show()

    # K-Means
    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Добавляем кластеры в DataFrame
    df["Cluster"] = clusters

    # Вывод количества элементов в каждом кластере
    print(df["Cluster"].value_counts())

    # Средние значения признаков по кластерам
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=data.feature_names)
    print("Координаты центров кластеров:")
    print(cluster_centers)

    # Группировка данных по кластерам
    cluster_analysis = df.groupby("Cluster").mean()
    print("Средние значения признаков по кластерам:")
    print(cluster_analysis)

    # PCA для визуализации
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Визуализация кластеров
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title("Кластеры вин (K-Means)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

#task1_3()



def task1_4():
    data = pd.read_csv("Data/Wholesale customers data.csv")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    inertia = []
    range_n_clusters = range(1, 10)

    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    plt.plot(range_n_clusters, inertia, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal Clusters")
    plt.show()

    # K-Means
    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)

    # Добавление кластеров в исходный DataFrame
    data["Cluster"] = clusters

    # Сколько объектов в каждом кластере
    print(data["Cluster"].value_counts())

    # Средние значения расходов по кластерам
    cluster_analysis = data.groupby("Cluster").mean()
    print("Средние значения расходов по кластерам:")
    print(cluster_analysis)

    # PCA для уменьшения размерности
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Визуализация кластеров
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title("Кластеры клиентов (K-Means)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

#task1_4()

 

def task2_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    df = pd.DataFrame(X, columns=feature_names)

    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Построение дендрограммы
    linked = linkage(X_scaled, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked, truncate_mode='lastp', p=30)
    plt.title("Дендрограмма")
    plt.xlabel("Samples")
    plt.ylabel("Euclidean Distance")
    plt.show()

    # Применение агломеративной кластеризации
    optimal_clusters = 3
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, metric='euclidean', linkage='ward')
    clusters = agg_clustering.fit_predict(X_scaled)

    # Добавление кластеров в DataFrame
    df['Cluster'] = clusters
    print(df['Cluster'].value_counts())

    # Оценка качества кластеризации
    ari = adjusted_rand_score(y, clusters)  # ARI (Adjusted Rand Index)
    ami = adjusted_mutual_info_score(y, clusters)  # AMI (Adjusted Mutual Information)

    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Adjusted Mutual Information (AMI): {ami}")

    # PCA для снижения размерности
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Визуализация кластеров
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title("Кластеры (Иерархическая кластеризация)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

    # Визуализация фактических меток
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
    plt.title("Фактические метки")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Class")
    plt.show()

#task2_1()



def task2_2():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    data = newsgroups.data
    target = newsgroups.target
    target_names = newsgroups.target_names

    print(f"Количество документов: {len(data)}")
    print(f"Количество категорий: {len(target_names)}")
    print(f"Пример текста:\n{data[0]}")

    # Настройка векторизатора TF-IDF
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')  # Ограничиваем число признаков для эффективности
    X_tfidf = vectorizer.fit_transform(data)

    print(f"Размерность векторизованных данных: {X_tfidf.shape}")

    # Используем подвыборку для дендрограммы (например, 200 документов)
    sample_size = 200
    X_sample = X_tfidf[:sample_size].toarray()

    # Построение дендрограммы
    linked = linkage(X_sample, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked, truncate_mode='lastp', p=20)
    plt.title("Дендрограмма (подвыборка)")
    plt.xlabel("Documents")
    plt.ylabel("Euclidean Distance")
    plt.show()

    # Применение агломеративной кластеризации
    optimal_clusters = 20
    agg_clustering = AgglomerativeClustering(
        n_clusters=optimal_clusters, metric='euclidean', linkage='ward'
    )
    clusters = agg_clustering.fit_predict(X_tfidf.toarray())

    # Сравнение с исходными метками
    df = pd.DataFrame({"Cluster": clusters, "Category": target})
    print(df.groupby("Cluster")["Category"].value_counts())

    # PCA для уменьшения размерности
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_tfidf.toarray())

    # Визуализация кластеров
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=10)
    plt.title("Кластеры (Иерархическая кластеризация)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

    # Оценка качества кластеризации
    ari = adjusted_rand_score(target, clusters)
    ami = adjusted_mutual_info_score(target, clusters)

    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Adjusted Mutual Information (AMI): {ami}")

#task2_2()



def task2_3():
    movies_path = "Data/movies_metadata.csv"
    ratings_path = "Data/ratings_small.csv"

    movies = pd.read_csv(movies_path, low_memory=False)
    ratings = pd.read_csv(ratings_path)

    # Извлечение жанров из столбца 'genres'
    def extract_genres(genres_str):
        try:
            genres = ast.literal_eval(genres_str)
            return [genre['name'] for genre in genres]
        except:
            return []

    movies['genres_list'] = movies['genres'].apply(extract_genres)

    # Приведение столбца 'id' в movies к числовому типу
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

    movies.dropna(subset=['id'], inplace=True)

    # Приведение типов для объединения
    movies['id'] = movies['id'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)

    # Расчет среднего рейтинга для каждого фильма
    average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)

    # Объединение данных о фильмах с данными о рейтингах
    movies = movies.merge(average_ratings, left_on='id', right_on='movieId', how='inner')

    # Оставляем только необходимые столбцы
    movies = movies[['title', 'genres_list', 'average_rating']]
    movies = movies[movies['genres_list'].map(len) > 0]

    # Преобразование жанров в бинарный вид
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(movies['genres_list']), columns=mlb.classes_)

    # Добавляем бинарные жанры к данным
    movies = pd.concat([movies, genres_encoded], axis=1)

    # Оставляем только числовые столбцы для кластеризации
    features = ['average_rating'] + list(mlb.classes_)
    data = movies[features]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Удаление строк с некорректными значениями
    valid_indices = ~np.isnan(data_scaled).any(axis=1) & ~np.isinf(data_scaled).any(axis=1)
    movies = movies[valid_indices].reset_index(drop=True)
    data_scaled = data_scaled[valid_indices]

    # Построение дендрограммы
    linked = linkage(data_scaled, method='ward')

    plt.figure(figsize=(15, 7))
    dendrogram(linked, truncate_mode='lastp', p=30)
    plt.title("Дендрограмма фильмов")
    plt.xlabel("Samples")
    plt.ylabel("Euclidean Distance")
    plt.show()

    # Применение агломеративной кластеризации
    optimal_clusters = 5  # Число кластеров из дендрограммы
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
    clusters = agg_clustering.fit_predict(data_scaled)

    # Добавляем кластеры в данные
    movies['Cluster'] = clusters

    # Смотрим количество фильмов в каждом кластере
    print("\nКоличество фильмов в каждом кластере:")
    print(movies['Cluster'].value_counts())

    # Выбираем только числовые столбцы для расчёта среднего значения
    numerical_columns = movies.select_dtypes(include=['number']).columns
    cluster_analysis = movies.groupby('Cluster')[numerical_columns].mean()

    print("\nСредние значения характеристик для каждого кластера:")
    print(cluster_analysis)

    # Частота жанров в каждом кластере
    genre_cluster_distribution = movies.groupby('Cluster')[mlb.classes_].mean()
    print("\nЧастота жанров по кластерам:")
    print(genre_cluster_distribution)

    # PCA для уменьшения размерности до 2D
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Визуализация кластеров
    plt.figure(figsize=(10, 7))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=10)
    plt.title("Кластеры фильмов (Иерархическая кластеризация)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

#task2_3()



def task2_4():
    columns = [
        "Area", "Perimeter", "Compactness", "KernelLength", "KernelWidth",
        "AsymmetryCoefficient", "KernelGrooveLength", "Type"
    ]

    data = pd.read_csv("Data/seeds_dataset.txt", delim_whitespace=True, names=columns)

    # Удаляем столбец "Type" (содержит метки классов)
    features = data.drop(columns=["Type"])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)

    # Построение дендрограммы
    # Построение иерархической кластеризации
    linked = linkage(data_scaled, method="ward")

    # Визуализация дендрограммы
    plt.figure(figsize=(15, 7))
    dendrogram(linked, truncate_mode="lastp", p=30)
    plt.title("Дендрограмма (Seeds Dataset)")
    plt.xlabel("Samples")
    plt.ylabel("Euclidean Distance")
    plt.show()

    # Применение агломеративной кластеризации
    optimal_clusters = 3  # Предположим, что оптимальное число кластеров равно 3
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage="ward")
    clusters = agg_clustering.fit_predict(data_scaled)

    # Добавляем кластеры в исходный DataFrame
    data["Cluster"] = clusters

    # Анализ кластеров
    print("\nКоличество объектов в каждом кластере:")
    print(data["Cluster"].value_counts())

    # Средние значения признаков для каждого кластера
    cluster_analysis = data.groupby("Cluster").mean()
    print("\nСредние значения признаков по кластерам:")
    print(cluster_analysis)

    # Визуализация кластеров (с помощью PCA)
    # Уменьшение размерности до 2D для визуализации
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Визуализация кластеров
    plt.figure(figsize=(10, 7))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap="viridis", s=50)
    plt.title("Кластеры семян (Иерархическая кластеризация)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

#task2_4()



def task3_1():
    iris = load_iris()
    X = iris.data  # Признаки
    y = iris.target  # Метки классов

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Применение DBSCAN
    # Настройка гиперпараметров DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)

    # Добавление кластеров в DataFrame
    data = pd.DataFrame(X, columns=iris.feature_names)
    data['Cluster'] = clusters
    data['True_Label'] = y

    # Сравнение результатов
    ari = adjusted_rand_score(y, clusters)
    ami = adjusted_mutual_info_score(y, clusters)
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Adjusted Mutual Information (AMI): {ami}")

    # Визуализация результатов
    # Преобразование данных в двумерное пространство с помощью PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Визуализация кластеров
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=50)
    plt.title("Кластеры (DBSCAN)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

    # Визуализация истинных меток классов
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=50)
    plt.title("Истинные метки классов")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="True Label")
    plt.show()

#task3_1()

def task3_2():
    X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
    data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    data["True Label"] = y

    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50)
    plt.title("Исходные данные (Moons)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Применение DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)

    # Добавление кластеров в DataFrame
    data["Cluster"] = clusters

    # Проверка распределения точек по кластерам
    print("\nРаспределение точек по кластерам:")
    print(data["Cluster"].value_counts())

    # Визуализация кластеров, найденных DBSCAN
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="viridis", s=50)
    plt.title("Кластеры, выделенные DBSCAN")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Cluster")
    plt.show()

    # Анализ шумовых точек
    noise_points = data[data["Cluster"] == -1]
    print("\nКоличество шумовых точек:")
    print(len(noise_points))

    # Визуализация шумовых точек отдельно
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="viridis", s=50, alpha=0.5)
    plt.scatter(
        noise_points["Feature 1"], noise_points["Feature 2"],
        c="red", label="Шумовые точки", edgecolors="black"
    )
    plt.title("Шумовые точки (DBSCAN)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

#task3_2()

