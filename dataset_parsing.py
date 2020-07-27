import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

if __name__ == '__main__':
    """ФУНКЦИЯ ДЛЯ СПИСКА БОГОВ"""
    def list_gods(file):
        spisok = file.readlines()
        for line in spisok:
            parts = line.split(', ')
            return parts[:-1]

    """ДЛЯ СЧЕТЧИКА МАТЧЕЙ В ДАТАСЕТЕ"""
    def enum(file):
        c = file.readlines()
        for line in c:
            schet = line.split()
            return int(schet[-1])

    """ДЛЯ ССЫЛОК МАТЧА В ДАТАСЕТЕ"""
    def number_match(file):
        link_match = []
        spisok = file.readlines()
        for line in spisok:
            parts = line.split()
            for i in range(0, len(parts)):
                link_match.append(parts[i])
        return link_match

    """ДЛЯ ИГРОКОВ"""
    def players(file):
        players_list = []
        spisok = file.readlines()
        for line in spisok:
            l = line.split('\n')
            players_list.append(l[0])
        return players_list

    UserAgent().chrome
    page = 'https://smite.guru'
    response = requests.get(page, headers={'User-Agent': UserAgent().chrome})
    html = response.content
    soup = BeautifulSoup(html, 'html.parser')

    path = open('name_gods.txt', 'r')
    path2 = open('D:/model_param/new_dataset_info/training_players.txt', 'r', encoding="utf-8")
    gods_list = list_gods(path)
    l_gods = {}
    for word in gods_list:
        if word not in l_gods:
            l_gods[word] = len(l_gods)

    player_list = players(path2)
    players_l = {}
    for i in range(0, len(player_list)):
        players_l[player_list[i]] = len(players_l)

    """СПИСОК МАТЧЕЙ(ССЫЛКИ)"""
    n_match = number_match(open('D:/model_param/new_dataset_info/new_dataset_all_links.txt', 'r'))
    print(n_match[-1])

    """поиск на странице последних матчей игроков с большим ELO"""
    objs = soup.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['recent-player'])
    e = enum(open('D:/model_param/new_dataset_info/schet_dop_data.txt', 'r'))  # счечтик матчей

    for obj in objs:
        # поиск среди тех матчей режим Conquest
        name_game = str(obj.find('div', attrs={'class': 'recent-player__updated'}).text).split()[1]
        # ссылка = номер матча
        match_link = obj.find('a', attrs={'class': 'recent-player__match'}).get('href')
        # проверка
        if name_game == 'Conquest' and match_link not in n_match:
            n_match.append(match_link)
            page = 'https://smite.guru' + str(match_link)
            resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
            html1 = resp.content
            soup1 = BeautifulSoup(html1, 'html.parser')
            # поиск 2 таблиц с данными 2 команд:КД, золото и т.п
            teams = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
            win_loss = [0, 0]
            pick = []
            pk = 0
            for team in range(0, 2):
                names_players = teams[team].findAll(lambda tag: tag.name == 'a' and tag.get('class') == ['row__player__name'])
                for j in range(0, len(names_players)):
                    name_player = '-'.join(names_players[j].get('href').split('/')[2].split('-')[1:])
                    if name_player in player_list:
                        pk += 1
            print(pk, page)
            if pk == 0:
                print(pk)
                print('CORRECT')
                for team in range(0, 2):
                    name_gods = teams[team].findAll(
                        lambda tag: tag.name == 'div' and tag.get('class') == ['row__player'])
                    for n_god in range(0, 5):
                        name_god = name_gods[n_god].find('img').get('alt')
                        if name_god in l_gods:
                            number_god = l_gods[name_god]
                            pick.append(number_god)
                        if name_god not in l_gods:
                            number_god = 'error'
                            pick.append(number_god)
                            print('error')
                            print(name_god)
                            print(match_link)
                        if name_god == 'Zhong Kui':
                            print(l_gods[name_god])
                            print(match_link)
                            print(name_god)
                    w_l = teams[team].find('div', 'match-table').get('class')[1]
                    if w_l == 'win':
                        win_loss[team] = 1
                    elif w_l == 'loss':
                        win_loss[team] = 0

                if 'error' in pick:
                    print('ERROR')

                if 'error' not in pick:
                    k = 0
                    for i in range(0, 5):
                        if pick[i] in pick[5:10]:
                            k += 1
                    if k == 0:
                        print(pick)
                        print()
                        for j in range(0, len(pick)):
                            with open('D:/model_param/new_dataset_info/valid.txt', 'a') as outFile:
                                outFile.write(str(pick[j]) + ' ')
                        with open('D:/model_param/new_dataset_info/valid.txt', 'a') as outFile:
                            outFile.write(str(win_loss[0]) + ' ' + str(win_loss[1]) + '\n')
                        e += 1
                        with open('D:/model_param/new_dataset_info/schet_dop_data.txt', 'a') as outFile:
                            outFile.write(' ' + str(e))
                        print(e)

                        with open('D:/model_param/new_dataset_info/valid_links.txt', 'a') as outFile:
                            outFile.write(str(match_link) + '\n')
                        with open('D:/model_param/new_dataset_info/new_dataset_all_links.txt', 'a') as outFile:
                            outFile.write(str(match_link) + '\n')
            print('_____________________________________________')




