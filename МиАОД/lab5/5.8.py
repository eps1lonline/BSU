import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

data = pd.read_csv("C:/Users/nikit/Desktop/Tweets.csv")

# Проверка данных
print(data.head())

# Инициализация стеммера, лемматизатора и стоп-слов
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Функция для предобработки текста
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление стоп-слов
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Стемминг
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    # Лемматизация
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Применение предобработки к текстовым данным
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Векторизация с использованием CountVectorizer
count_vectorizer = CountVectorizer()
count_vectors = count_vectorizer.fit_transform(data['cleaned_text'])

# Векторизация с использованием TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(data['cleaned_text'])

# Проверка результатов
print("Count Vectorizer Shape:", count_vectors.shape)
print("TFIDF Vectorizer Shape:", tfidf_vectors.shape)

# Пример векторизованных данных
print("Пример Count Vectorizer:")
print(count_vectors.toarray()[:5])
print("Пример TFIDF Vectorizer:")
print(tfidf_vectors.toarray()[:5])
