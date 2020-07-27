import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
'''СОЗДАНИЕ СПИСКА ИНФОРМАЦИИ 22000 МАТЧЕЙ'''
info_match = []
file = open('upd_qonq.txt', 'r')
spisok = file.readlines()
for line in spisok:
    list_match = []
    parts = line.split()
    for i in parts:
        list_match.append(int(i))
    info_match.append(list_match)
print(info_match)

name_gods = {}
file = open('dateInfo.txt', 'r')
spisok2 = file.readlines()
for line in spisok2:
    parts = line.split(', ')
    for i in parts:
        if i not in name_gods:
            name_gods[i] = len(name_gods)
print(name_gods)

winrate = np.zeros(101)
match = np.zeros(101)

for i in range(0, 10000):
    if info_match[i][10] == 1:
        for j in range(0, 5):
            winrate[info_match[i][j]] += 1
    if info_match[i][11] == 1:
        for j in range(5, 10):
            winrate[info_match[i][j]] += 1
    for j in range(0, 10):
        match[info_match[i][j]] += 1
print(winrate)
print(match)
winrate2 = np.zeros(101)
for i in range(0, 101):
    if match[i] != 0:
        winrate2[i] = (100 * winrate[i]) / match[i]
    if match[i] == 0:
        print(i, match[i])
print(winrate2)


def baseline(start, size_sample, winrate):
    baseline= 0
    for i in range(start, size_sample):
        winrate_team1 = 0
        winrate_team2 = 0
        win_irl = 0
        win_bas = 0
        for j in range(0, 5):
            winrate_team1 += winrate[info_match[i][j]]
        for j in range(5, 10):
            winrate_team2 += winrate[info_match[i][j]]
        winrate_team1 = winrate_team1 / 5
        winrate_team2 = winrate_team2 / 5
        if info_match[i][11] == 1:
            win_irl = 1
        if winrate_team1 < winrate_team2:
            win_bas = 1
        if win_irl == win_bas:
            baseline += 1
    return (baseline/(size_sample - start)) * 100


print(baseline(0, 8000, winrate2))
print(baseline(8000, 9000, winrate2))
print(baseline(9000, 10000, winrate2))
