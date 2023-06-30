import pandas as pd

def anonymize_dataframe(df, columns):
    for column in columns:
        df[column] = df[column].apply(lambda x: 'REDACTED')
    return df

data = pd.DataFrame({'Name': ['Alice', 'Bob', 'Carol'],
                     'Age': [25, 30, 35],
                     'Address': ['123 Main St', '456 Elm St', '789 Oak St']})

anonymized_data = anonymize_dataframe(data, ['Name', 'Address'])
print(anonymized_data)
