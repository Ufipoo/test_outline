import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import visdom

info_match = []
file = open('dataset.txt', 'r')
spisok = file.readlines()
win1_tr = 0
win2_tr = 0
win1_va = 0
win2_va = 0
win1_te = 0
win2_te = 0
for line in spisok:
    list_match = []
    parts = line.split()
    for i in parts:
        list_match.append(int(i))
    info_match.append(list_match)

print(info_match)

for i in range(0, 20000):
    if info_match[i][10] == 1:
        win1_tr += 1
    if info_match[i][11] == 1:
        win2_tr += 1
print(win1_tr, win2_tr)

for i in range(20000, 21000):
    if info_match[i][10] == 1:
        win1_va += 1
    if info_match[i][11] == 1:
        win2_va += 1
print(win1_va, win2_va)

for i in range(21000, 22000):
    if info_match[i][10] == 1:
        win1_te += 1
    if info_match[i][11] == 1:
        win2_te += 1
print(win1_te, win2_te)

name_gods = {}
file = open('name_gods.txt', 'r')
spisok = file.readlines()
for line in spisok:
    parts = line.split(', ')
    for i in parts:
        if i not in name_gods:
            name_gods[i] = len(name_gods)
print(name_gods)

print(len(name_gods))

vec = np.zeros(len(name_gods))
for i in range(0, 22000):
    for word in range(0, 10):
        vec[info_match[i][word]] += 1
print(vec)


vec1 = np.zeros(len(name_gods))
for i in range(0, 20000):
    for word in range(0, 10):
        vec1[info_match[i][word]] += 1
print(vec1)

vec2 = np.zeros(len(name_gods))
for i in range(20000, 21000):
    for word in range(0, 10):
        vec2[info_match[i][word]] += 1
print(vec2)

vec3 = np.zeros(len(name_gods))
for i in range(21000, 22000):
    for word in range(0, 10):
        vec3[info_match[i][word]] += 1
print(vec3)


vec4 = np.zeros(len(name_gods))
for i in range(0, len(name_gods)):
    if vec[i] != 0:
        vec4[i] = '{:.3f}'.format((vec1[i] * 100) / vec[i])
print('TRAIN')
print(vec4)


vec5 = np.zeros(len(name_gods))
for i in range(0, len(name_gods)):
    if vec[i] != 0:
        vec5[i] = '{:.3f}'.format((vec2[i] * 100) / vec[i])
print('VALIDATION')
print(vec5)

vec6 = np.zeros(len(name_gods))
for i in range(0, len(name_gods)):
    if vec[i] != 0:
        vec6[i] = '{:.3f}'.format((vec3[i] * 100) / vec[i])
print('TEST')
print(vec6)

yb = np.zeros(101)
#print(yb.size)

def make_bow_vector(sentence):
    vec = np.linspace(0, 100, 101)
    vec2 = []
    team1 = [0, 1, 0]
    team2 = [0, 0, 1]
    not_god = [1, 0, 0]
    for i in range(0, 101):
        if vec[i] in sentence[0:5]:
            vec2.append(team1) 
        if vec[i] in sentence[5:10]:
            vec2.append(team2)
        if vec[i] not in sentence[0:10]:
            vec2.append(not_god)
    print(vec2)
    vec2 = np.concatenate(np.array(vec2))
    return vec2

print('_______________')
"""
team1 = torch.tensor([0., 1., 0.])
team2 = torch.tensor([0., 0., 1.])
print(team2)
data2 = []
for i in range(0, 3):
    instance = make_bow_vector(np.array(info_match[i][:10]))
    data2.append(instance)
god_kol = np.zeros(101)
data2 = torch.Tensor(data2)
print(data2)
for i in range(0, 3):
    er = np.reshape(data2[i], (101, 3))
    print(er)
    for j in range(0, len(er)):
        if er[j][1] == 1:

            god_kol[j] += 1
            print(er[j], team1)
            print(j)
            print('____________')
        if er[j][2] == 1:

            god_kol[j] += 1
            print(er[j], team2)
            print(j)
            print('__________')
for i in range(0, len(god_kol)):
    print(i, god_kol[i])
"""
"""
train = np.array([58.058, 57.441, 58.582, 58.621, 59.167, 59.158, 62.09,  55.418, 61.224, 59.657, 59.378, 59.705, 59.165,
                  56.753, 62.68,  58.739, 59.073, 58.325, 58.342, 56.969, 57.643, 58.597, 59.104, 55.238, 60.264, 60.533,
                  56.983, 58.68,  59.042, 60.5,   58.925, 54.214, 62.025, 58.989, 57.426, 59.626, 59.393, 59.71,  58.292,
                  59.789, 60.3,   61.228, 59.428, 59.072, 60.534, 60.258, 58.009, 59.612, 60.656, 57.239, 59.014, 58.52,
                  63.272, 59.49,  60.706, 56.954, 59.551, 58.545, 56.563, 61.023, 61.527, 57.5,   60.523, 59.974, 58.641,
                  57.75,  59.325, 61.146, 56.836, 59.504, 58.917, 58.52,  61.467, 58.517, 60.041, 58.028, 59.886, 57.534,
                  58.733, 55.889, 59.826, 56.239, 58.877, 59.006, 58.484, 57.922, 56.857, 59.804, 57.765, 57.835, 60.377,
                  62.939, 57.003, 62.687, 59.531, 58.366, 59.039, 0.,     59.274, 62.364, 60.61])"""
"""meen_train = np.sum(train)/train.size
print('Accuracy train')
print(meen_train)
print(train.size)"""
"""
valid = np.array([57.851, 57.107, 59.208, 59.24,  59.329, 58.581, 61.154, 55.728, 60.204, 59.034, 59.6,   59.478, 58.802,
                  56.753, 62.62,  57.736, 59.369, 59.202, 58.015, 56.698, 59.043, 58.426, 59.104, 56.19,  60.543, 61.067,
                  55.307, 57.963, 59.203, 60.375, 58.925, 55.809, 61.772, 60.815, 58.02,  59.813, 58.444, 60.,    59.53,
                  58.947, 59.738, 62.725, 59.596, 59.494, 60.237, 60.961, 58.787, 60.582, 62.09,  57.744, 59.354, 59.036,
                  63.272, 59.065, 60.82,  57.194, 60.534, 58.545, 56.205, 61.209, 61.671, 58.462, 60.523, 59.061, 58.473,
                  57.452, 58.73,  63.376, 57.422, 59.835, 58.121, 56.502, 61.777, 59.148, 59.016, 58.945, 59.744, 57.323,
                  60.103, 55.336, 59.652, 55.897, 59.239, 59.783, 58.979, 58.571, 57.714, 61.928, 57.7,   58.12,  60.549,
                  62.3,   59.244, 62.572, 59.105, 58.463, 59.718,  0.,    59.037, 62.545, 60.61])
"""
"""meen_valid = np.sum(valid)/valid.size
print('Accuracy validation')
print(meen_valid)
print(valid.size)"""
"""
summa = ((np.sum(train)/101)+(np.sum(valid)/101))/2
print(summa)
summa = (train+valid)/2
print(summa.size)
itog = np.sum(summa)/summa.size
print(itog)
"""
"""
test = np.array([55.963, 63.277, 61.644, 53.285, 57.339, 58., 49.367, 61.905, 63.077, 49.367,
                 58.252, 59.398, 55.882, 52.857, 57.377, 55.769, 56.471, 59.69, 58.182, 61.176,
                 55.399, 62.879, 66.216, 62., 57.143, 57.447, 48.387, 62.921, 59.406, 60.377,
                 66.197, 49.153, 52.727, 52.381, 58.416, 64.384, 64.789, 63.158, 59.813, 52.632,
                 47.887, 60.526, 61.039, 55.752, 60.87, 51.402, 56.627, 52.941, 44.068, 61.538,
                 71.264, 61.111, 65.625, 54.945, 63.38, 53.333, 61.842, 50.649, 56.311, 54.962,
                 58.696, 68., 55.882, 60.92, 54.098, 56.111, 51.786, 57.143, 61.404, 49.333,
                 52.874, 64.706, 51.304, 60.87, 62.406, 61.111, 50., 58.416, 55.952, 54.032,
                 63.235, 60.563, 56.338, 51.136, 42.857, 60.36, 53.659, 55.072, 52.941, 56.79,
                 54.321, 60.526, 48.864, 61.062, 57.798, 53.333, 58.065, 60.241, 50., 59.596])
meen_test = np.sum(test)/test.size
print('Accuracy test')
print(meen_test)
print(test.size)"""

"""
# Initialize

x = torch.Tensor([[[-0.0288,  0.0470,  0.0026, -0.0160,  0.0104, -0.0151],
                   [-0.1120,  0.0539,  0.0596, -0.0493,  0.0401,  0.0668],
                   [-0.1280,  0.0779,  0.1412, 0.0215, -0.0307,  0.0236],
                   [-0.0680,  0.0248,  0.1506, -0.0436,  0.0739, -0.0033]],
                  [[-0.0059,  0.0641,  0.1732, 0.0077,  0.0603, -0.0931],
                   [-0.0849, -0.0047,  0.1190, -0.0642,  0.1105, -0.0431],
                   [-0.1643,  0.0008, -0.0079, -0.0414,  0.0679, -0.0070],
                   [-0.0157,  0.0090,  0.0035, -0.0075,  0.0386, -0.0686]]])
print(x)
c = torch.Tensor(2,6)
b = torch.Tensor(2,6)
#print(c)
for i in range(0, 2):
    c[i][0] = x.pow(2).sum()
    #c[i] = x[i][0] + x[i][1]
   # b[i] = x[i][2] + x[i][3]
print(c)
print(b)
print('=========')
#print(torch.cat((c,b),1))
b = torch.arange(4 * 10 * 1024).view(4, 10, 1024)
c = torch.Tensor(4, 6)
d = torch.Tensor()
print(b)

print(torch.cat((torch.sum(b[:][:,0:5],1), torch.sum(b[:][:,5:10],1)), 1))"""
#print(torch.sum(b,1))
"""
import visdom
vis = visdom.Visdom()
vis.replay_log('D:/model_param/test14/graph')"""

"""vis = visdom.Visdom()  # new visdom that actually connects to the server
vis.replay_log('D:/model_param/test13/graph')"""

"""
def make_bow_vector2(sentence):
    vec = np.zeros(101)
    for word in range(0, 10):
        vec[sentence[word]] += 1
    for p in range(0, 101):
        if vec[p] == 0:
            vec[p] = -1
    print(vec)
    return vec

data2 = []
for i in range(0, 3):
    instance = make_bow_vector2(np.array(info_match[i][:10]))
    data2.append(instance)

data2 = torch.Tensor(data2)
print(data2)"""