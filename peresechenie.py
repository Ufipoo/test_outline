import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import sys
import time
import os

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
if __name__ == '__main__':
    print(sys.version)

    #UserAgent().chrome
    path1 = open('D:/model_param/new_dataset_info/training_players.txt', 'r', encoding="utf-8")
    path2 = open('links_match.txt', 'r')  # 5000-10 000


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


    def players(file):
        players_list = []
        spisok = file.readlines()
        for line in spisok:
            l = line.split('\n')
            players_list.append(l[0])
        return players_list
    o = 10.0
    r = np.zeros(np.int(o))
    print(r)

    player = players(open('D:/model_param/new_dataset_info/training_players.txt', 'r', encoding="utf-8"))
    print(player)

    link2 = number_match2(path2)
    print(link2)
    print('______________________')
    train_hero_v = np.zeros(77547)
    train_hero_t = np.zeros(77547)
    hero_valid = np.zeros(77547)
    hero_test  = np.zeros(77547)
    #v = matchs(path3)
    for i in range(20000, 21000):
        page = 'https://smite.guru' + str(link2[i])
        print(page, i)
        error_players = 0
        time.sleep(4)

        resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
        html1 = resp.content
        soup1 = BeautifulSoup(html1, 'html.parser')
        teams = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
        for team in range(0, 2):
            names_players = teams[team].findAll(
                lambda tag: tag.name == 'a' and tag.get('class') == ['row__player__name'])
            for j in range(0, len(names_players)):
                name_player = '-'.join(names_players[j].get('href').split('/')[2].split('-')[1:])
                if name_player:
                    if name_player in player:
                        print(name_player, player[player.index(name_player)])
                        error_players += 1
                        g = player.index(name_player)
                        if train_hero_v[g] == 0:
                            train_hero_v[g] += 1
                        hero_valid[g] += 1
                        print(name_player, player[g])

    print('TEST TEST TEST TEST')
    for i in range(21000, 22480):
        page = 'https://smite.guru' + str(link2[i])
        print(page, i)
        error_players = 0
        time.sleep(4)
        resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
        html1 = resp.content
        soup1 = BeautifulSoup(html1, 'html.parser')
        teams = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
        for team in range(0, 2):
            names_players = teams[team].findAll(
                lambda tag: tag.name == 'a' and tag.get('class') == ['row__player__name'])
            for j in range(0, len(names_players)):
                name_player = '-'.join(names_players[j].get('href').split('/')[2].split('-')[1:])
                if name_player:
                    if name_player in player:
                        print(name_player, player[player.index(name_player)])
                        error_players += 1
                        g = player.index(name_player)
                        if train_hero_t[g] == 0:
                            train_hero_t[g] += 1
                        hero_test[g] += 1
                        print(name_player, player[g])

    k_valid = 0
    k_test  = 0

    for j in range(0, 77547):
        if train_hero_v[j] == 1:
            k_valid += 1
        if train_hero_t[j] == 1:
            k_test += 1

    print('ИГРОКИ ПО ОДНОМУ МАССИВ')
    print(train_hero_v, train_hero_t)
    print('ИГРОКИ ВСЕГО СКОЛЬКО РАЗ МАССИВ')
    print(hero_valid, hero_test)
    print('ПО ОДНОМУ')
    print(k_valid, k_test)
    with open('D:/model_param/new_dataset_info/peresechnie_one.txt', 'a', encoding="utf-8") as outFile:
        outFile.write(str(train_hero_v) + '\n' + str(hero_valid))
    with open('D:/model_param/new_dataset_info/peresechnie_two.txt', 'a', encoding="utf-8") as outFile:
        outFile.write(str(train_hero_t) + '\n' + str(hero_test))

    max_v = np.max(hero_valid)
    max_t = np.max(hero_test)
    print(max_v, max_t)
    valid_peresechenie = np.zeros(int(max_v+1))
    test_peresechenie = np.zeros(int(max_v+1))


    for i in range(0, int(max_v)):
        for j in range(0, 77547):
            if hero_valid[j] == i:
                valid_peresechenie[i] += 1
    for i in range(0, int(max_t)):
        for j in range(0, 77547):
            if hero_test[j] == i:
                test_peresechenie[i] += 1

    print('ПО НЕСКОЛЬКО РАЗ')
    print(valid_peresechenie, test_peresechenie)




    '''print(train_hero)
    path2 = open('D:/model_param/players3.txt', 'r', encoding="utf-8")
    path3 = open('D:/model_param/players4.txt', 'r', encoding="utf-8")

    def players(file):
        players_list = []
        spisok = file.readlines()
        for line in spisok:
            l = line.split('\n')
            players_list.append(l[0])
        return players_list


    train_p = players(path1)
    valid_p = players(path2)
    test_p = players(path3)
    valid_zero = np.zeros(47270)
    test_zero = np.zeros(47270)
    obch = np.zeros(47270)
    print(valid_p)
    print(test_p)
    print('******************************')

    for i in range(0, 47270):
        for j in range(0, 6727):
            if train_p[i] == valid_p[j]:
                print(train_p[i], i, j)
                valid_zero[i] += 1
                obch[i] += 1

    print('______________________________')

    for i in range(0, 47270):
        for j in range(0, 7130):
            if train_p[i] == test_p[j]:
                print(train_p[i], i, j)
                test_zero[i] += 1
                obch[i] += 1

    print('################################')
    print('VALID')
    print(valid_zero)
    print('TEST')
    print(test_zero)
    print('TEAIN')
    print(obch)
    k_valid = 0
    k_test = 0
    k_obch = 0
    k = 0
    for i in range(0, 47270):
        if valid_zero[i] != 0:
            k_valid += 1
        if test_zero[i] != 0:
            k_test += 1
        if obch[i] == 2:
            k_obch += 1
        if obch[i] == 1:
            k += 1
    print('________________________________')
    print('________________________________')
    print('________________________________')
    print('VALID PERESECHENIE')
    print(k_valid)
    print('________________________________')
    print('TEST PERESECHENIE')
    print(k_test)
    print('________________________________')
    print('OBCH PERESECHENIE == 2')
    print(k_obch)
    print('________________________________')
    print('PERESECHENIE == 1')
    print(k)'''









