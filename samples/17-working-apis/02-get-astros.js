import requests

url = 'http://api.open-notify.org/astros.json'
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')
