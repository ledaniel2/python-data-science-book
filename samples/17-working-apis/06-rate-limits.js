import requests
import time

urls = ['https://api.example.com/data1', 'https://api.example.com/data2', 'https://api.example.com/data3']
pause_seconds = 2

for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')
    time.sleep(pause_seconds)
