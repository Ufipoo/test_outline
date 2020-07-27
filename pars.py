import requests
import bs4
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
#UserAgent().chrome
page = 'https://smite.guru/builds'
response = requests.get(page, headers={'User-Agent': UserAgent().chrome})
print(response)
for key, value in response.request.headers.items():
    print(key+": "+value)
html = response.content
soup = BeautifulSoup(html, 'html.parser')

print(soup.html.head.title.text)
objs = soup.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['champion__title'])
gods = {}


