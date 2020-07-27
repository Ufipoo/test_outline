import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import sys

def info_match(file, i): #выборка из матча одного
    spisok = file.readlines()
    parts = spisok[i].split()
    viborka = []
    for j in range(0, len(parts)):
        viborka.append(int(parts[j]))
    return viborka

def enum(file):
    c = file.readlines()
    for line in c:
        schet = line.split()
        return int(schet[0])

temp = []
'''ПРОВЕРКУ НА ПОВТОРЕНИЯ ГЕРОЕВ ПРОШЛО В UPD_CONQ'''
'''
for i in range(0, 5000):
    inf_m = info_match(open('upd_qonq.txt', 'r'), i)
    for j in range(0, 5):
        if inf_m[j] in inf_m[5:10]:
            if i not in temp:
                temp.append(i)
print(temp)
'''

def link_m(file):
    c = file.readlines()
    for line in c:
        parts = line.split(',')
        return parts
f = link_m(open('n_match.txt', 'r'))
print(f)
print(len(f))
'''СОБИРАЕМ ССЫЛКИ НОРМАЛЬНЫХ 5000 МАТЧЕЙ'''
'''
for i in range(5000, 10000):
    with open('link_upd_qonq.txt', 'a') as outFile:
        outFile.write(f[i])
    with open('link_upd_qonq.txt', 'a') as outFile:
            outFile.write('\n')
'''
'''СОБИРАЕМ ССЫЛКИ  КОТОРЫЕ НАДО ПРОВЕРИТЬ ВМЕСТЕ С ИНФОМАТЧ'''
'''
for i in range(0, 5000):
    with open('link_which_sort.txt', 'a') as outFile:
        outFile.write(f[i])
    with open('link_which_sort.txt', 'a') as outFile:
            outFile.write('\n')
'''

