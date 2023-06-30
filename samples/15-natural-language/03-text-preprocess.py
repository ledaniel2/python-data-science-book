text = 'Learn Python for Data Science!'
lowercase_text = text.lower()
print(lowercase_text)  # Output: learn python for data science!

import nltk
nltk.download('punkt')

text = 'Learn Python for Data Science!'
tokens = nltk.word_tokenize(text)
print(tokens)  # Output: ['Learn', 'Python', 'for', 'Data', 'Science', '!']

import string

text = 'Learn Python for Data Science!'
translator = str.maketrans('', '', string.punctuation)
no_punctuation_text = text.translate(translator)
print(no_punctuation_text)  # Output: Learn Python for Data Science

from nltk.corpus import stopwords
nltk.download('stopwords')

text = 'Learn Python for Data Science!'
tokens = nltk.word_tokenize(text.lower())
filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
print(filtered_tokens)  # Output: ['learn', 'python', 'data', 'science', '!']

from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

text = 'Learn Python for Data Science!'
tokens = nltk.word_tokenize(text.lower())

stemmed_tokens = [stemmer.stem(token) for token in tokens]
print('Stemmed tokens:', stemmed_tokens)  # Output: ['learn', 'python', 'for', 'data', 'scienc', '!']

lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print('Lemmatized tokens:', lemmatized_tokens)  # Output: ['learn', 'python', 'for', 'data', 'science', '!']
