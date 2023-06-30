import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

text = 'Learn Python for Data Science!'
preprocessed_text = preprocess_text(text)
print(preprocessed_text)  # Output: ['learn', 'python', 'data', 'science']
