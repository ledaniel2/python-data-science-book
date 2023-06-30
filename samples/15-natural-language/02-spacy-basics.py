import spacy

# Load a pre-trained language model
nlp = spacy.load('en_core_web_sm')  # Use: python -m spacy download en_core_web_sm

# Process the text using the language model
text = "spaCy is a modern and efficient library for natural language processing in Python. It's great for large-scale applications."
doc = nlp(text)

# Tokenization
word_tokens = [token.text for token in doc]
sent_tokens = [sent.text for sent in doc.sents]
print('Word tokens:', word_tokens)
print('Sentiment tokens:', sent_tokens)

# Part-of-speech tagging
tagged_tokens = [(token.text, token.pos_) for token in doc]
print('Tagged tokens:', tagged_tokens)

# Named entity recognition
entities = [(entity.text, entity.label_) for entity in doc.ents]
print('Named entities:', entities)
