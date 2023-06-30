import requests

api_key = 'your_api_key'
symbol = 'MSFT'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    time_series = data['Time Series (Daily)']
    for date, values in time_series.items():
        print(f'{date}: {values}')
else:
    print(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')
