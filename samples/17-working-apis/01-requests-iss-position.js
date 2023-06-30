import requests

# Define the API endpoint
url = 'http://api.open-notify.org/iss-now.json'

# Send a GET request to the API
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON data from the response
    data = response.json()

    # Extract the latitude and longitude of the ISS
    latitude = data['iss_position']['latitude']
    longitude = data['iss_position']['longitude']

    # Print the current location of the ISS
    print(f'The ISS is currently located at latitude {latitude} and longitude {longitude}.')
else:
    print(f'Error: Unable to retrieve data from the API (status code {response.status_code}).')
