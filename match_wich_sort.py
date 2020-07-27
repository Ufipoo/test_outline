import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


def number_match(file, i): #ссылки матчей
    spisok = file.readlines()
    return spisok[i]


print(number_match(open('link_upd_qonq.txt', 'r'),3))


def info_match(file, i): #выборка из матча одного
    spisok = file.readlines()
    parts = spisok[i].split()
    viborka = []
    for j in range(0, len(parts)):
        viborka.append(int(parts[j]))
    return viborka


def list_gods(file):
    spisok = file.readlines()
    for line in spisok:
        parts = line.split(', ')
        return parts


path = open('dateInfo.txt', 'r')
gods_list = list_gods(path)


l_gods = {}
for word in gods_list:
    if word not in l_gods:
        l_gods[word] = len(l_gods)
print(l_gods)


def enum(file): #колличество матчей
    c = file.readlines()
    for line in c:
        schet = line.split()
        return int(schet[0])


for i in range(3176, 5000):
    page = 'https://smite.guru' + str(number_match(open('link_which_sort.txt', 'r'), i))
    resp = requests.get(page, headers={'User-Agent': UserAgent().chrome})
    html1 = resp.content
    soup1 = BeautifulSoup(html1, 'html.parser')
    teams = soup1.findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['scrollable'])
    win_loss = [0, 0]
    proverka = []
    for team in range(0, 2):

        name_gods = teams[team].findAll(
            lambda tag: tag.name == 'div' and tag.get('class') == ['row__player'])
        builds = teams[team].findAll(lambda tag: tag.name == 'div' and tag.get('class') == ['build'])

        for n_god in range(0, 5):
            name_god = name_gods[n_god].find('img').get('alt')
            build = builds[n_god].findAll('div', 'build__holder')
            if name_god in l_gods:
                number_god = l_gods[name_god]
                proverka.append(number_god)
            else:
                proverka.append('error')

        w_l = teams[team].find('div', 'match-table').get('class')[1]
        if w_l == 'win':
            win_loss[team] = 1
        elif w_l == 'loss':
            win_loss[team] = 0
    proverka.append(win_loss[0])
    proverka.append(win_loss[1])
    print(str(i) + ' = ' + str(proverka))

    inf_m = info_match(open('InfoMatch.txt', 'r'), i)
    print(inf_m)
    if proverka == inf_m:
        print('god')
        '''k = 0
        for j in range(0, 5):
            if proverka[j] in proverka[5:10]:
                k += 1
        if k == 0:'''
        for good in range(0, len(proverka)):
            with open('old_qonq.txt', 'a') as outFile:
                outFile.write(str(proverka[good]) + ' ')
        with open('old_qonq.txt', 'a') as outFile:
            outFile.write('\n')


    elif 'error' in proverka:
        '''with open('not_n11m.txt', 'a') as outFile:
            outFile.write(str(i) + ' ------ ' + str(n_match[i]) + '\n')'''
        print('unkwn good')
        with open('link_error.txt', 'a') as outFile:
            outFile.write(str(i+1) + '------' + str(number_match(open('link_which_sort.txt', 'r'), i)) + '\n')

    else:
        '''k = 0
        for j in range(0, 5):
            if proverka[j] in proverka[5:10]:
                k += 1
        if k == 0:'''
        for good in range(0, len(proverka)):
            with open('old_qonq.txt', 'a') as outFile:
                outFile.write(str(proverka[good]) + ' ')
        with open('old_qonq.txt', 'a') as outFile:
            outFile.write('\n')
        print('not error, but not in InfoMatch')
    print('_____________________________________________')


