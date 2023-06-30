import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Tokenization
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLTK is a powerful library for natural language processing in Python. It's great for beginners and experts alike."
word_tokens = word_tokenize(text)
sent_tokens = sent_tokenize(text)
print('Word tokens:', word_tokens)
print('Sentiment tokens:', sent_tokens)

# Part-of-speech tagging
from nltk import pos_tag

tagged_tokens = pos_tag(word_tokens)
print('Tagged tokens:', tagged_tokens)

# Lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in word_tokens]
print('Lemmatized tokens:', lemmatized_tokens)
