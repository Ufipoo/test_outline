import numpy as np
from scipy.optimize import linprog
import time
# функция для открытия файлов и считывания данных
def file_open(link_file, j):
    file = open(link_file, 'r')
    list_nutr = []
    spisok = file.readlines()
    parts = spisok[j].split()
    for i in parts:
        list_nutr.append(int(i))
    return list_nutr
# список пит.веществ
list_nutrients = ['proteins.txt', 'carbohydrates.txt', 'mineral salts.txt', 'fats.txt']


for i in range(1):
    start = time.time()
    # сбор данных
    c = []  # price
    A_1 = []  # nutrients in food
    b_1 = []  # norm nutrients
    c.append(file_open('price/rainbow fish.txt', i)[0])
    c.append(file_open('price/gold fish.txt', i)[0])
    for j in list_nutrients:
        A_1.append((file_open('nutrients/' + str(j), i)))
        b_1.append((file_open('norma/' + str(j), i)[0]))
    print('price RAINBOW FISH and GOLD FISH: ' + str(c))
    print('PROTEINS in food: ' + str((A_1[0])))
    print('CARBOHYDRATES in food: ' + str(A_1[1]))
    print('MINERAL SALTS in food: ' + str(A_1[2]))
    print('FATS in food: ' + str(A_1[3]))
    print('norm nutrients in food: ' + str(b_1))
    print('________RESULTS________')

    res = linprog(c, np.array(A_1)*(-1), np.array(b_1)*(-1))
    stop = time.time()
    print("Time:", stop - start)
    print(res)
    # запись результатов в файл
    with open('res.txt', 'a') as outFile:
        outFile.write("Time:" + str(stop - start) + '\n' + str(res) + '\n' + '___________' + '\n')
    print('_______________________')

    # формирование матрицы из векторов, которые входят в базис
    for p in range(1,3):
        cb = []
        cb.append(file_open('price/rainbow fish.txt', i)[0])
        cb.append(file_open('price/gold fish.txt', p)[0])
        x = res['x']
        m1 = []
        n = 0
        for j in range(2):
            if x[j] != 0:  # вектор учавствует в базисе
                n += 1
                for k in range(0, len(A_1)):
                    m1.append(A_1[k][j] * (-1))
        # если доп вектора входят в базис
        xd = res['slack']
        for j in range(len(xd)):
            if xd[j] != 0:
                cb.append(0)
                n += 1
                v = np.zeros(4)
                v[j] = 1
                for k in range(4):
                    m1.append(v[k] * (-1))

        m = np.reshape(m1, (4, n)).T
        #print('matrica: \n', m)
        # базис
        b = np.linalg.inv(m)
        print('basis: \n', b)

        # формируем часть симплекс таблицы
        s_table = []
        n = 0
        for j in range(2):
            v = []
            for k in range(0, len(A_1)):
                s_table.append(A_1[k][j] * (-1))

        s_table = np.reshape(s_table, (2, 4)).T
        delt = []
        for j in range(2):
            de = np.dot(b, s_table[:, j].T)
            desum = sum(cb * de) - cb[j]  # ищем оценки
            delt.append(desum)
        dop_p = np.eye(4)

        for j in range(4):
            de = np.dot(b, dop_p[:, j].T)
            desum = sum(cb * de)  # ищем оценки
            delt.append(desum)
        print('оценки: \n', delt)
        f = np.dot(b, np.array(b_1) * (-1))
        f = np.sum(cb * f)
        print('решение: \n', f)
        # проверям на оптимальность

        for j in range(len(delt)):
            if delt[j] <= 0:
                if j == len(delt) - 1:
                    print('При изменение цены (',cb[:2] , ') решение остается оптимальным: ', f)
                    with open('res.txt', 'a') as outFile:
                        outFile.write('При изменение цены (' + str(cb[:2]) + ') решение остается оптимальным: ' + str(f)
                                      + '\n' + '________________________' + '\n')
                    print('_______________________')

                    break
            if delt[j] > 0:
                res = linprog(cb[:2], np.array(A_1) * (-1), np.array(b_1) * (-1))
                print('При изменение цены (', cb[:2], ') решение допустимое, но не оптимальное, => ищем новый оптимум: \n', res)
                with open('res.txt', 'a') as outFile:
                    outFile.write('При изменение цены (' + str(cb[:2]) + ')  решение допустимое, но не оптимальное, => ищем новый оптимум: '
                                  + '\n' + str(res) + '\n' + '________________________' + '\n')
                break
        print('__________________________________')




