import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Example text data
text_data = pd.DataFrame({'text': ['I love data science', 'Machine learning is amazing', 'Python is great']})

# Text vectorization - Bag-of-Words
bow_vectorizer = CountVectorizer()
text_bow = bow_vectorizer.fit_transform(text_data['text'])

# Text vectorization - TF-IDF
tfidf_vectorizer = TfidfVectorizer()
text_tfidf = tfidf_vectorizer.fit_transform(text_data['text'])

print(text_tfidf)
