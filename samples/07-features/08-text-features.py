import pandas as pd

data = pd.DataFrame({'text_data': ['One man went to mow, went to mow a meadow', 'Whether the weather be fine, whether the weather be not']})
# Calculate the length of each text entry
data['text_length'] = data['text_data'].str.len()

# Count the number of words in each text entry
data['word_count'] = data['text_data'].str.split().str.len()

# Count the number of unique words in each text entry
data['unique_word_count'] = data['text_data'].apply(lambda x: len(set(x.lower().split())))

print(data.drop('text_data', axis=1))
