import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Load the dataset
url = 'https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary.csv'
data = pd.read_csv(url, nrows=250, encoding='iso-8859-1')

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess the text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic tokens
    words = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stopwords.words('english')]
    
    return words

# Preprocess the news articles
data['preprocessed_text'] = data['text'].apply(preprocess_text)

# Create a dictionary from the preprocessed text
dictionary = Dictionary(data['preprocessed_text'])

# Filter out infrequent and overly frequent words
dictionary.filter_extremes(no_below=20, no_above=0.5)

# Convert the preprocessed text to bag-of-words format
corpus = [dictionary.doc2bow(text) for text in data['preprocessed_text']]

# Train an LDA model
num_topics = 10
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42, passes=10)

# Print the top words for each topic
num_words = 10
for i, topic in enumerate(lda_model.print_topics(num_words=num_words)):
    print(f'Topic {i + 1}: {topic}')

def get_topic_distribution(text, dictionary, lda_model):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Convert the preprocessed text to bag-of-words format
    bow = dictionary.doc2bow(preprocessed_text)

    # Get the topic distribution for the input text
    topic_distribution = lda_model.get_document_topics(bow)

    return topic_distribution

# Example usage of the get_topic_distribution function
new_article = 'Officials release statistics showing an increased number of women in Government, including ministers.'
topic_distribution = get_topic_distribution(new_article, dictionary, lda_model)
print(f'Topic distribution: {topic_distribution}')
