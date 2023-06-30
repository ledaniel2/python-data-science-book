import requests
from requests.auth import HTTPBasicAuth

url = 'https://api.example.com/data'
username = 'your_username'
password = 'your_password'

response = requests.get(url, auth=HTTPBasicAuth(username, password))

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')
