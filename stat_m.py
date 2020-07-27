import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import sys

if __name__ == '__main__':
    print(sys.version)

    UserAgent().chrome
    page = 'https://smite.guru'

    response = requests.get(page, headers={'User-Agent': UserAgent().chrome})
    print(response)

    html = response.content
    soup = BeautifulSoup(html, 'html.parser')

    print(soup.html.head.title.text)

    path = open('dateInfo.txt', 'r')


    def list_gods(file):
        spisok = file.readlines()
        for line in spisok:
            parts = line.split(', ')
            return parts


    gods_list = list_gods(path)
    print(gods_list)
    l_gods = {}
    for word in gods_list:
        if word not in l_gods:
            l_gods[word] = len(l_gods)

    print(l_gods)


    def enum(file):
        c = file.readlines()
        for line in c:
            schet = line.split()
            return int(schet[0])


    def number_match(file):
        spisok = file.readlines()
        for line in spisok:
            parts = line.split(',')
            return parts


    n_match = number_match(open('n_match.txt', 'r'))
    print(n_match)

    # поиск на странице последних матчей игроков с большим ELO
    objs = soup.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['recent-player'])

    for obj in objs:
        # поиск среди тех матчей режим Conquest
        name_game = str(obj.find('div', attrs={'class': 'recent-player__updated'}).text).split()[1]
        # ссылка = номер матча
        match_link = obj.find('a', attrs={'class': 'recent-player__match'}).get('href')
        e = enum(open('Schet.txt', 'r'))
        # проверка
        if e < 5000:
            # print(enumerator)

            if name_game == 'Conquest' and match_link not in n_match:
                e += 1
                with open('Schet.txt', 'w') as outFile:
                    outFile.write(str(e))
                with open('n_match.txt', 'a') as outFile:
                    outFile.write(match_link + ',')
                print(name_game + match_link)
                n_match.append(match_link)
                page = 'https://smite.guru' + str(match_link)
                resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
                html1 = resp.content
                soup1 = BeautifulSoup(html1, 'html.parser')
                # поиск 2 таблиц с данными 2 команд:КД, золото и т.п
                teams = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
                win_loss = [0, 0]
                # print(win_loss)
                for team in range(0, 2):
                    # print('+')
                    # поиск сатистики по богам
                    name_gods = teams[team].findAll(
                        lambda tag: tag.name == 'div' and tag.get('class') == ['row__player'])
                    builds = teams[team].findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['build'])

                    for n_god in range(0, 5):
                        name_god = name_gods[n_god].find('img').get('alt')
                        build = builds[n_god].findAll('div', 'build__holder')
                        if name_god in l_gods:
                            number_god = l_gods[name_god]

                            with open('Builds.txt', 'a') as outFile:
                                outFile.write(str(number_god) + ' ')

                            for item in build:
                                if item.find('img'):
                                    name_item = item.find('img').get('alt')
                                    with open('Builds.txt', 'a') as outFile:
                                        outFile.write(name_item + ', ')

                            with open('Builds.txt', 'a') as outFile:
                                outFile.write('\n')
                            print(name_god + ' ' + str(number_god))
                            with open('InfoMatch.txt', 'a') as outFile:
                                outFile.write(str(number_god) + ' ')

                    w_l = teams[team].find('div', 'match-table').get('class')[1]
                    if w_l == 'win':
                        win_loss[team] = 1
                    elif w_l == 'loss':
                        win_loss[team] = 0

                with open('InfoMatch.txt', 'a') as outFile:
                    outFile.write(str(win_loss[0]) + ' ' + str(win_loss[1]) + '\n')
                # print(win_loss)
                print('_____________________________________________')

