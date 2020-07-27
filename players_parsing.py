import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import sys
if __name__ == '__main__':
    print(sys.version)

    UserAgent().chrome
    path1 = open('links_match.txt', 'r')
    path3 = open('dataset.txt', 'r')
    path6 = open('D:/model_param/new_dataset_info/training_players.txt', 'r', encoding="utf-8")
    path7 = open('D:/model_param/new_dataset_info/valid_links.txt', 'r', encoding="utf-8")
    path8 = open('D:/model_param/new_dataset_info/player_valid.txt', 'r', encoding="utf-8")
    path9 = open('D:/model_param/new_dataset_info/new_dataset_all_links.txt', 'r', encoding="utf-8")


    def number_match(file):
        link_match = []
        spisok = file.readlines()
        for line in spisok:
            parts = line.split()
            for i in range(0, len(parts)):
                link_match.append(parts[i])
        return link_match

    def number_match2(file):
        link_match = []
        spisok = file.readlines()
        for line in spisok:
            parts = line.split(',')
            for i in range(0, len(parts)):
                if parts[i] != '0':
                    link_match.append(parts[i])
        return link_match

    def players2(file):
        players_list = []
        k = []
        links = []
        spisok = file.readlines()
        for line in spisok:
            l = line.split(', ')
            if len(l) == 3:
                players_list.append(l[0])
                links.append(l[1])
                k.append(l[2].split('\n')[0])
            if len(l) == 2:
                players_list.append(l[0])
                links.append(l[1].split(',')[0])
                k.append(1)
            if len(l) == 1:
                players_list.append(l[0].split(',')[0])
                links.append(' ')
                k.append(0)
        return players_list, k, links

    def matchs(file):
        info_match = []
        spisok = file.readlines()
        for line in spisok:
            list_match = []
            parts = line.split()
            for i in parts:
                list_match.append(int(i))
            info_match.append(list_match)
        return info_match

    def list_gods(file):
        spisok = file.readlines()
        for line in spisok:
            parts = line.split(', ')
            return parts[:-1]

    def enum(file):
        c = file.readlines()
        for line in c:
            schet = line.split()
            return int(schet[-1])

    def players(file):
        players_list5 = []
        spisok = file.readlines()
        for line in spisok:
            l = line.split('\n')
            players_list5.append(l[0])
        return players_list5


    v = matchs(path3)
    error_list = ['1-', '2-', '3-', '4-', '5-', '6-', '7-', '8-', '9-', '10-']
    players_list = players(path6)  # игроки тренировчногой выборки
    player = {} # словарь игрково тренировочной выборки
    for i in range(0, len(players_list)):
        if players_list[i] not in player:
            player[players_list[i]] = len(player)


    all_links = number_match(path9)
    valid_igroki, k_str, links = players2(path8)

    for i in range(0, len(valid_igroki)):
        if valid_igroki[i] in players_list:
            print('ИГРОК ИЗ ВАЛДИАЦИИ ЕСТЬ В ТРЕНИРОВКЕ')
            print(player[valid_igroki[i]], valid_igroki[i])
    print('______________________')
    path = open('name_gods.txt', 'r')
    gods_list = list_gods(path)
    #print(gods_list)
    l_gods = {}
    for word in gods_list:
        if word not in l_gods:
            l_gods[word] = len(l_gods)
    print('______________________')


    e = int(enum(open('D:/model_param/new_dataset_info/schet.txt', 'r', encoding="utf-8")))
    print(e)
    if k_str[e] != 0:
        search_name = valid_igroki[e]
        link = links[e]
        t = int(k_str[e])
        if search_name not in players_list:
            for i in range(1, t+1):
                page = link + str(i)
                print(page, i)
                resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
                html1 = resp.content
                soup1 = BeautifulSoup(html1, 'html.parser')
                match = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['top'])

                for j in range(0, len(match)):
                    m = match[j].find('div', attrs={'class': 'title'}).text.split(' ')[1]
                    if m == 'Conquest':
                        page2 = match[j].find('a', attrs={'class': 'sub'}).get('href')
                        if page2 not in all_links:
                            print(m)
                            print(page2)
                            page3 = 'https://smite.guru' + str(page2)
                            resp2 = requests.get(page3, headers={'User-Agent': UserAgent().chrome})
                            html2 = resp2.content
                            soup2 = BeautifulSoup(html2, 'html.parser')
                            # поиск 2 таблиц с данными 2 команд:КД, золото и т.п
                            teams = soup2.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
                            win_loss = [0, 0]
                            pick = []
                            pk = 0
                            u = []
                            # print(win_loss)
                            for team in range(0, 2):
                                # print('+')
                                # поиск сатистики по богам
                                names_players = teams[team].findAll(
                                    lambda tag: tag.name == 'a' and tag.get('class') == ['row__player__name'])
                                for j in range(0, len(names_players)):
                                    name_player = '-'.join(names_players[j].get('href').split('/')[2].split('-')[1:])
                                    if name_player and name_player not in error_list:
                                        u.append(name_player)
                                    if name_player in players_list:
                                        pk += 1
                                       # print(player[name_player], name_player)
                           # print(pk, page3)
                            if pk == 0:
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
                                            print(page3)
                                        if name_god == 'Zhong Kui':
                                            print(l_gods[name_god])
                                            print(page3)
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
                                        for q in range(0, len(pick)):
                                            with open('D:/model_param/new_dataset_info/valid.txt', 'a') as outFile:
                                                outFile.write(str(pick[q]) + ' ')
                                        with open('D:/model_param/new_dataset_info/valid.txt', 'a') as outFile:
                                            outFile.write(str(win_loss[0]) + ' ' + str(win_loss[1]) + '\n')
                                        with open('D:/model_param/new_dataset_info/valid_links.txt', 'a') as outFile:
                                            outFile.write(str(page2) + '\n')
                                        with open('D:/model_param/new_dataset_info/new_dataset_all_links.txt',
                                                  'a') as outFile:
                                            outFile.write(str(page2) + '\n')
                                        all_links.append(page2)
                                        for y in range(0, len(u)):
                                            print(u[y])
                                            if u[y] not in valid_igroki:
                                                with open('D:/model_param/new_dataset_info/player_valid.txt', 'a',
                                                          encoding="utf-8") as outFile:
                                                    outFile.write(str(u[y]) + ', ' + '\n')
                                                valid_igroki.append(u[y])
                            print('_____________________________________________')
    e += 1
    with open('D:/model_param/new_dataset_info/schet.txt', 'a',
              encoding="utf-8") as outFile:
        outFile.write(str(e) + ' ')




