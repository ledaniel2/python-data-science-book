import requests
from retry import retry

@retry(tries=5, delay=2, backoff=2)
def get_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')

urls = ['https://api.example.com/data1', 'https://api.example.com/data2', 'https://api.example.com/data3']

for url in urls:
    try:
        data = get_data(url)
        print(data)
    except Exception as e:
        print(e)
