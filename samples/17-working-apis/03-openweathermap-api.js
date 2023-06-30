import requests

api_key = 'your_api_key'
# Latitude and longitude of New York (40.7128 degrees N, 74.0060 degrees W)
lat = 40.7128
lon = -74.0060
url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}'

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')
