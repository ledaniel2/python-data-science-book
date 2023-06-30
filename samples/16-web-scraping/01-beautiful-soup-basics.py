import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/Web_scraping'
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

title = soup.find('title').text
description = soup.find('div', {'class': 'mw-parser-output'}).find('p').text

print(f'Title: {title}')
print(f'Description: {description}')
