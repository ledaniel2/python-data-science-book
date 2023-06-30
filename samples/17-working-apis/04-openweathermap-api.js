import requests

def handle_api_error(response):
    if response.status_code != 200:
        try:
            error_message = response.json().get('message', 'Unknown error')
        except ValueError:
            error_message = 'Unknown error'
        print(f'Error: Unable to retrieve data from the API (status code {response.status_code}, message: {error_message}).')
        return False
    return True

api_key = 'your_api_key'
city = 'London'
url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'

response = requests.get(url)

if handle_api_error(response):
    data = response.json()
    print(data)
