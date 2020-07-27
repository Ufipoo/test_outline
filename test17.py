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
import random
print(torch.cuda.is_available())
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.__version__)
print('__________')
#graph = 'D:/model_param/test17/graph'
vis = visdom.Visdom()
#vis.replay_log(graph)
'''СОЗДАНИЕ СПИСКА ИНФОРМАЦИИ 10000 МАТЧЕЙ'''
info_match = []
file = open('upd_qonq.txt', 'r')
spisok = file.readlines()
for line in spisok:
    list_match = []
    parts = line.split()
    for i in parts:
        list_match.append(int(i))
    info_match.append(list_match)
print(len(info_match))

name_gods = {}
file = open('dateInfo.txt', 'r')
spisok2 = file.readlines()
for line in spisok2:
    parts = line.split(', ')
    for i in parts:
        if i not in name_gods:
            name_gods[i] = len(name_gods)
print(name_gods)

vv = np.zeros(len(name_gods))
for i in range(0, 8000):
    for word in range(0, 10):
        vv[info_match[i][word]] += 1
print(vv)

gods_val = np.zeros(len(name_gods))
for i in range(8000, 9000):
    for word in range(0, 10):
        gods_val[info_match[i][word]] += 1
print(gods_val)
gods_test = np.zeros(len(name_gods))
for i in range(9000, 10000):
    for word in range(0, 10):
        gods_test[info_match[i][word]] += 1
print(gods_test)

label_to_ix = {'[1 0]': 0, '[0 1]': 1}

'''ПОДГОТОВКА ДАННЫХ К СБОРУ В ДАТАСЕТ'''

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
    vec2 = np.concatenate(np.array(vec2))
    return np.array(vec2)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[str(label)]])


data2 = []
data3 = []
for i in range(0, 10000):
    instance = make_bow_vector(np.array(info_match[i][:10]))
    target = make_target(np.array(info_match[i][10:]), label_to_ix)
    data3.append(target)
    data2.append(instance)


data2 = torch.Tensor(data2)
data3 = torch.Tensor(data3)
print(data2)
print(data3)

x_train = data2[:8000]
y_train = data3[:8000]
x_valid = data2[8000:9000]
y_valid = data3[8000:9000]
x_test = data2[9000:10000]
y_test = data3[9000:10000]

'''ПАРАМЕТРЫ, МОДЕЛЬ'''
batch_size = 4
#lr = 0.0001
# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(303, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


model = NeuralNet().to(device)

checkpoint = torch.load('D:/model_param/test13/itr_1288000')
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()

#update_lr(optimizer, 0.01)
model.train()
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

test_ds = TensorDataset(x_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size)

'''НАЧАЛАО ОБУЧЕНИЯ'''
global_iteration1 = 0
global_iteration2 = 0
val_loss = 0
irl_target_val, net_target_val = [], []
toch2 = 0
IRL_target, NET_target = [], []
m_loss = 0
toch = 0

def make_vec(inst):
    er = np.reshape(inst.cpu(), (101, 3))
    for p in range(0, len(er)):
        if er[p][1] == 1:
            er[p][1] = 0
            er[p][2] = 1
        elif er[p][2] == 1:
            er[p][1] = 1
            er[p][2] = 0
    er = np.concatenate(np.array(er))
    return er


def make_label(lab):
    if lab == 0:
        lab = 1
    elif lab == 1:
        lab = 0
    return lab

"""
while True:
    model.train()
    for i, (instance, labels) in enumerate(train_dl):
        print(labels)
        vec = []
        l = []
        buf = []
        for j in range(0, 4):
            argument = random.random()
            print(argument)
            if argument >= 0.5:
                buf.append(1)
                vec.append(make_vec(instance[j]))
                l.append(labels[j])
                labels[j] = make_label(labels[j])
            else:
                vec.append(np.array(instance[j]))
                l.append(labels[j])
                buf.append(0)
        print(labels)
        instance = instance.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        log_probs = model(instance)
        print(log_probs)
        loss = loss_function(log_probs, labels)
        m_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pre_y = torch.max(log_probs, 1)[1]

        NET_target.append(pre_y.cpu().data.numpy())
        IRL_target.append(labels.cpu().data.numpy())

        vec = []
        for j in range(0, 4):
            if buf[j] == 1:
                vec.append(make_vec(instance[j]))
                labels[j] = make_label(labels[j])
        print(labels)

        global_iteration1 += 1
        print(global_iteration1)

        if global_iteration1 % 10 == 0:
            m_loss2 = m_loss / 10
            m_loss = float(0)
            NET_target2 = np.array(NET_target)
            IRL_target2 = np.array(IRL_target)
            NET_target, IRL_target = [], []
            toch = float((IRL_target2 == NET_target2).astype(int).sum()) / float(IRL_target2.size)
            vis.line(Y=np.array([m_loss2.item()]), X=np.array([global_iteration1]), win='plot_loss',
                     update='append', opts=dict(title='Loss train', xlabel='Iteration', ylabel='Loss'))
            vis.line(Y=np.array([toch]), X=np.array([global_iteration1]), win='plot_acc', update='append',
                     opts=dict(title='Accuracy train', xlabel='Iteration', ylabel='Loss'))


    print('VALIDATION')
    model.eval()
    with torch.no_grad():
        for i, (instance, labels) in enumerate(valid_dl):
            instance = instance.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            log_probs = model(instance)
            loss = loss_function(log_probs, labels)
            val_loss += loss
            pre_y = torch.max(log_probs, 1)[1]
            net_target_val.append(pre_y.cpu().data.numpy())
            irl_target_val.append(labels.cpu().data.numpy())

            global_iteration2 += 1

            if global_iteration2 % 1000 == 0:
                mean_loss = val_loss / 1000
                val_loss = float(0)
                net_target2 = np.array(net_target_val)
                irl_target2 = np.array(irl_target_val)
                irl_target_val, net_target_val = [], []
                toch2 = float((irl_target2 == net_target2).astype(int).sum()) / float(irl_target2.size)
                vis.line(Y=np.array([mean_loss.item()]), X=np.array([global_iteration2]),
                         win='plot_loss_val', update='append',
                         opts=dict(title='Loss validation', xlabel='Iteration', ylabel='Loss'))
                vis.line(Y=np.array([toch2]), X=np.array([global_iteration2]), win='plot_acc_val',
                         update='append',
                         opts=dict(title='Accuracy validation', xlabel='Iteration', ylabel='Loss'))
                torch.save({
                    'iteration': global_iteration2,
                    'model_state_dict': model.state_dict(),
                    'loss_train': mean_loss,
                    'accuracy_train': toch2})

"""


god_ac = np.zeros(101)
god_ac_v = np.zeros(101)
god_ac_t = np.zeros(101)
"""
NET_target, IRL_target = [], []
with torch.no_grad():
    for i, (instance, labels) in enumerate(train_dl):
        instance = instance.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        log_probs = model(instance)
        loss = loss_function(log_probs, labels)
        pre_y = torch.max(log_probs, 1)[1]

        for j in range(0, 4):
            if pre_y[j] == labels[j]:
                er = np.reshape(instance[j].cpu(), (101, 3))
                for p in range(0, len(er)):
                    if er[p][1] == 1:
                        god_ac[p] += 1

                    if er[p][2] == 1:
                        god_ac[p] += 1

        NET_target.append(pre_y.cpu().data.numpy())
        IRL_target.append(labels.cpu().data.numpy())
print("___________________")
NET_target2 = np.array(NET_target)
IRL_target2 = np.array(IRL_target)
accua = float((IRL_target2 == NET_target2).astype(int).sum()) / float(IRL_target2.size)
print('TRAIN ACCURACY')
print(accua)

for j in range(0, god_ac.size):
    if god_ac[j] != 0:
        god_ac[j] = '{:.3f}'.format((god_ac[j] / vv[j]) * 100)
with open('D:/model_param/test13/train_accuracy_1.txt', 'a') as outFile:
    outFile.write(str(god_ac))
print(god_ac)
print(np.sum(god_ac)/101)
"""
NET_target, IRL_target = [], []
with torch.no_grad():
    for i, (instance, labels) in enumerate(valid_dl):
        instance = instance.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        log_probs = model(instance)
        loss = loss_function(log_probs, labels)
        pre_y = torch.max(log_probs, 1)[1]

        for j in range(0, 4):
            if pre_y[j] == labels[j]:
                er = np.reshape(instance[j].cpu(), (101, 3))
                for p in range(0, len(er)):
                    if er[p][1] == 1:
                        god_ac_v[p] += 1

                    if er[p][2] == 1:
                        god_ac_v[p] += 1

        NET_target.append(pre_y.cpu().data.numpy())
        IRL_target.append(labels.cpu().data.numpy())
NET_target2 = np.array(NET_target)
IRL_target2 = np.array(IRL_target)
accua = float((IRL_target2 == NET_target2).astype(int).sum()) / float(IRL_target2.size)
print('VALID ACCURACY')
print(accua)

for j in range(0, god_ac_v.size):
    if god_ac_v[j] != 0:
        god_ac_v[j] = '{:.3f}'.format((god_ac_v[j] / gods_val[j]) * 100)
with open('D:/model_param/test13/valid_accuracy_1.txt', 'a') as outFile:
    outFile.write(str(god_ac_v))
print(god_ac_v)
print(np.sum(god_ac_v)/101)

NET_target, IRL_target = [], []
with torch.no_grad():
    for i, (instance, labels) in enumerate(test_dl):
        instance = instance.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        log_probs = model(instance)
        loss = loss_function(log_probs, labels)
        pre_y = torch.max(log_probs, 1)[1]

        for j in range(0, 4):
            if pre_y[j] == labels[j]:
                er = np.reshape(instance[j].cpu(), (101, 3))
                for p in range(0, len(er)):
                    if er[p][1] == 1:
                        god_ac_t[p] += 1

                    if er[p][2] == 1:
                        god_ac_t[p] += 1

        NET_target.append(pre_y.cpu().data.numpy())
        IRL_target.append(labels.cpu().data.numpy())
print("___________________")
NET_target2 = np.array(NET_target)
IRL_target2 = np.array(IRL_target)
accua = float((IRL_target2 == NET_target2).astype(int).sum()) / float(IRL_target2.size)
print('TEST ACCURACY')
print(accua)

for j in range(0, god_ac_t.size):
    if god_ac_t[j] != 0:
        god_ac_t[j] = '{:.3f}'.format((god_ac_t[j] / gods_test[j]) * 100)
with open('D:/model_param/test13/test_accuracy_1.txt', 'a') as outFile:
    outFile.write(str(god_ac_t))
print(god_ac_t)
print(np.sum(god_ac_t)/101)
"""_________________________________________________________________"""








