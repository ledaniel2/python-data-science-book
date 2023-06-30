import requests

api_key = 'your_api_key'
address = '1600+Amphitheatre+Parkway,+Mountain+View,+CA'
url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    location = data['results'][0]['geometry']['location']
    print(location)
else:
    print(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')
