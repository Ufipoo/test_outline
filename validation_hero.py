import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import sys
import time
import os
if __name__ == '__main__':
    print(sys.version)

    UserAgent().chrome
    path1 = open('links_match.txt', 'r')  # 5000-10 000
    path2 = open('D:/model_param/new_dataset_info/training_players.txt', 'r', encoding="utf-8")
    path3 = open('D:/model_param/new_dataset_info/valid_naproverku_links.txt', 'r', encoding="utf-8")
    #path4 = open('D:/model_param/new_dataset_info/test_naproverku_links.txt', 'r', encoding="utf-8")

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

    def enum(file):
        c = file.readlines()
        for line in c:
            schet = line.split()
            return int(schet[-1])

    def vec(file, n_match):
        info_match = []
        spisok = file.readlines()
        return spisok[n_match]

    player_list = players(path2)
    print('______________________')
    #print(player_list)
    link1 = number_match(path3)
    #print(link1)
    #link2 = number_match(path4)
    e_v = enum(open('D:/model_param/new_dataset_info/schet_valid.txt', 'r'))
    e_dop = enum(open('D:/model_param/new_dataset_info/schet_dop_data.txt', 'r'))
    print(e_v, e_dop)
    path5 = open('dataset.txt', 'r')
    players_l = {}
    for i in range(0, len(player_list)):
        players_l[player_list[i]] = len(players_l)
    #print(players_l)


    """for i in range(0, 20000):
        with open('D:/model_param/new_dataset_info/training_links.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(link1[i]) + '\n')
        with open('training_links.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(link1[i]) + '\n')"""

    '''link = number_match2(path1)
    for i in range(20000, 22481):
        with open('D:/model_param/new_dataset_info/valid_naproverku_links.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(link[i]) + '\n')
        with open('valid_naproverku_links.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(link[i]) + '\n')'''

    """for i in range(21000, 22000):
        with open('D:/model_param/new_dataset_info/test_naproverku_links.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(link1[i]) + '\n')
        with open('test_naproverku_links.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(link1[i]) + '\n')"""

    page = 'https://smite.guru' + str(link1[e_v])
    print(page, e_v)
    resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
    html1 = resp.content
    soup1 = BeautifulSoup(html1, 'html.parser')
    teams = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
    k = 0
    for team in range(0, 2):
        names_players = teams[team].findAll(
            lambda tag: tag.name == 'a' and tag.get('class') == ['row__player__name'])
        for j in range(0, len(names_players)):
            name_player = '-'.join(names_players[j].get('href').split('/')[2].split('-')[1:])
            if name_player in player_list:
                k += 1
    if k == 0:
        print('CORRECT')
        with open('D:/model_param/new_dataset_info/valid.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(vec(path5, e_v)))
        with open('D:/model_param/new_dataset_info/valid_links.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(page) + '\n')
        with open('D:/model_param/new_dataset_info/schet_dop_data.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(' ' + str(e_dop+1))
    else:
        print('ANCORRECT')

    e_v += 1
    with open('D:/model_param/new_dataset_info/schet_valid.txt', 'a') as outFile:
        outFile.write(' ' + str(e_v))

""" page = 'https://smite.guru' + str(link2[e_t])
    print(page, e_t)
    resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
    html1 = resp.content
    soup1 = BeautifulSoup(html1, 'html.parser')
    teams = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
    k = 0
    for team in range(0, 2):
        names_players = teams[team].findAll(
            lambda tag: tag.name == 'a' and tag.get('class') == ['row__player__name'])
        for j in range(0, len(names_players)):
            name_player = '-'.join(names_players[j].get('href').split('/')[2].split('-')[1:])
            if name_player in player_list:
                print(name_player, players_l[name_player])
                k += 1
                print('+')
    if k == 0:
        print('CORRECT')
        with open('D:/model_param/new_dataset_info/test.txt', 'a', encoding="utf-8") as outFile:
            outFile.write(str(vec(path5, e_v)) + '\n')
    e_t += 1
    with open('D:/model_param/new_dataset_info/schet_test.txt', 'a', encoding="utf-8") as outFile:
        outFile.write(' ' + str(e_t))"""

