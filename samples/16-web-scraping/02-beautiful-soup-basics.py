import requests
from bs4 import BeautifulSoup

url = 'https://slashdot.org'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

articles = soup.find_all('span', {'class': 'story-title'})

for article in articles:
    title = article.text.strip()
    print(f'Title: {title}')
