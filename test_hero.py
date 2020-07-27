import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import sys
import time
import os
if __name__ == '__main__':
    print(sys.version)

    UserAgent().chrome
    path1 = open('D:/model_param/new_dataset_info/training_players.txt', 'r', encoding="utf-8")
    path2 = open('D:/model_param/new_dataset_info/valid_links.txt', 'r', encoding="utf-8")


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

    def players(file):
        players_list = []
        spisok = file.readlines()
        for line in spisok:
            l = line.split('\n')
            players_list.append(l[0])
        return players_list
    player = {}

    link1 = number_match(path2)
    #print(link1)
    player_list = players(path1)
    print('______________________')
    for i in range(0, len(player_list)):
        if player_list[i] not in player:
            player[player_list[i]] = len(player)
    print(player)
    print(player['Truantsy'])
    player['Truantsy'] -= 500
    print(player['Truantsy'])
    print(player)
    print(max(player.values()))
    #print(player_list)

    """for i in range(0, 18):
        page = link1[i]
        print(page, i)
        time.sleep(1)
        resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
        html1 = resp.content
        soup1 = BeautifulSoup(html1, 'html.parser')
        teams = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
        y = 0
        for team in range(0, 2):
            names_players = teams[team].findAll(
                lambda tag: tag.name == 'a' and tag.get('class') == ['row__player__name'])
            for j in range(0, len(names_players)):
                name_player = '-'.join(names_players[j].get('href').split('/')[2].split('-')[1:])
                if name_player:
                    po = players(open('D:/model_param/new_dataset_info/training_players.txt', 'r', encoding="utf-8"))
                    if name_player not in po:
                        y += 1
                        print(name_player)
                        print('+')
                        player[name_player] = len(player)
        if y == 10:
            print(y, page)

    print(player)"""


