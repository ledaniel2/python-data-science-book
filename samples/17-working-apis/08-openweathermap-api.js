import requests

api_key = 'your_api_key'
# The following line contains a space, which is encoded by requests.get()
city = 'New York'
url = 'http://api.openweathermap.org/data/2.5/weather'

response = requests.get(url, [('q', city), ('appid', api_key)])

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')
