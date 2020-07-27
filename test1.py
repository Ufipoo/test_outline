import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import visdom
import random

print(torch.cuda.is_available())
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.__version__)
print('__________')
graph = 'D:/model_param/test21/graph'
vis = visdom.Visdom()
#vis.replay_log(graph)

'''СОЗДАНИЕ СПИСКА ИНФОРМАЦИИ 10000 МАТЧЕЙ'''
info_match = []
file = open('dataset.txt', 'r')
spisok = file.readlines()
for line in spisok:
    list_match = []
    parts = line.split()
    for i in parts:
        list_match.append(int(i))
    info_match.append(list_match)
print(len(info_match))

name_gods = {}
file = open('name_gods.txt', 'r')
spisok2 = file.readlines()
for line in spisok2:
    parts = line.split(', ')
    for i in parts:
        if i not in name_gods:
            name_gods[i] = len(name_gods)
print(name_gods)

label_to_ix = {'[1 0]': 0, '[0 1]': 1}


'''ПОДГОТОВКА ДАННЫХ К СБОРУ В ДАТАСЕТ'''


def make_vector(sentence):
    vec_hero = []
    for i in range(0, 10):
        hero = np.zeros(105)
        hero[sentence[i]] = 1
        vec_hero.append(hero)
    return vec_hero


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[str(label)]])


data2 = []
data3 = []
for i in range(0, 22000):
    instance = make_vector(info_match[i][:10])
    target = make_target(np.array(info_match[i][10:]), label_to_ix)
    data3.append(target)
    data2.append(instance)


data2 = torch.Tensor(data2)
data3 = torch.Tensor(data3)
#print(data2)
#print(data3)

x_train = data2[:20000]
y_train = data3[:20000]
x_valid = data2[20000:21000]
y_valid = data3[20000:21000]
x_test = data2[21000:22000]
y_test = data3[21000:22000]

'''ПАРАМЕТРЫ, МОДЕЛИ'''
batch_size = 4


def make_vec(inst):
    temp = torch.Tensor(10,105)
    for i in range(0, 5):
        temp[i] = inst[5+i]
        temp[5+i] = inst[i]
    return temp


def sum_conc(x):
    return torch.cat((torch.sum(x[:][:,0:5],1), torch.sum(x[:][:,5:10],1)), 1)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(105, 1024)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = sum_conc(out)
        out = out.to(device)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


model = NeuralNet().to(device)
print(model)

checkpoint = torch.load('D:/model_param/test25/itr_87000')
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
loss_function = nn.CrossEntropyLoss()
model.train()

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

'''НАЧАЛАО ОБУЧЕНИЯ'''
global_iteration1 = 6962500
global_iteration2 = 343000
val_loss = 0
irl_target_val, net_target_val = [], []
toch2 = 0
IRL_target, NET_target = [], []
m_loss = 0
m_loss2 = 0
toch = 0


while True:
    model.train()
    for i, (instance, labels) in enumerate(train_dl):

        temp = torch.Tensor(4,10,105)
        temp_labels = torch.Tensor(4)
        for j in range(0, 4):
            argument = random.random()
            #print(argument)
            if argument >= 0.5:
                temp[j] = make_vec(instance[j])
                if labels[j] == 0:
                    temp_labels[j] = 1
                elif labels[j] == 1:
                    temp_labels[j] = 0
            else:
                temp[j] = instance[j]
                if labels[j] == 0:
                    temp_labels[j] = 0
                elif labels[j] == 1:
                    temp_labels[j] = 1

        #instance = instance.to(device)
        #labels = labels.to(device=device, dtype=torch.int64)
        temp = temp.to(device)
        temp_labels = temp_labels.to(device=device, dtype=torch.int64)

        log_probs = model(temp)  # обучение

        loss = loss_function(log_probs, temp_labels)  # функция потери
        m_loss += loss
        optimizer.zero_grad()
        loss.backward()  # обратное распрастранение ошибки
        optimizer.step()

        pre_y = torch.max(log_probs, 1)[1]

        NET_target.append(pre_y.cpu().data.numpy())
        IRL_target.append(temp_labels.cpu().data.numpy())

        global_iteration1 += 1

        if global_iteration1 % 500 == 0:
            print('train' + str(global_iteration1))
            m_loss2 = m_loss / 500
            m_loss = float(0)
            NET_target2 = np.array(NET_target)
            IRL_target2 = np.array(IRL_target)
            NET_target, IRL_target = [], []
            toch = float((IRL_target2 == NET_target2).astype(int).sum()) / float(IRL_target2.size)
            vis.line(Y=np.array([m_loss2.item()]), X=np.array([global_iteration1]), win='plot_loss',
                     update='append', opts=dict(title='Loss train', xlabel='Iteration', ylabel='Loss'))
            vis.line(Y=np.array([toch]), X=np.array([global_iteration1]), win='plot_acc', update='append',
                     opts=dict(title='Accuracy train', xlabel='Iteration', ylabel='Accuracy'))


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
                print('VALIDATION' + str(global_iteration2))
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
                         opts=dict(title='Accuracy validation', xlabel='Iteration', ylabel='Accuracy'))
                torch.save({
                    'iteration': global_iteration2,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_train': m_loss2,
                    'accuracy_train': toch
                }, 'D:/model_param/test25/itr_' + str(global_iteration2))



"""ПРОВЕРКА САМОЙ ЛУЧШЕЙ МОДЕЛИ НА ТОЧНОСТЬ ПО КАЖДОМУ ГЕРОЮ"""
"""
info_match2 = []
file = open('upd_qonq.txt', 'r')
spisok = file.readlines()
for line in spisok:
    list_match = []
    parts = line.split()
    for i in parts:
        list_match.append(int(i))
    info_match2.append(list_match)
print(len(info_match2))

gods_val = np.zeros(101)
for i in range(8000, 9000):
    for word in range(0, 10):
        gods_val[info_match2[i][word]] += 1
print(gods_val)


data4 = []
data5 = []
for i in range(8000, 9000):
    instance = make_vector(info_match2[i][:10])
    target = make_target(np.array(info_match2[i][10:]), label_to_ix)
    data5.append(target)
    data4.append(instance)


data4 = torch.Tensor(data4)
data5 = torch.Tensor(data5)

x_valid = data4[:1000]
y_valid = data5[:1000]

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)


def accuracy_heroes(model_check):
    model = NeuralNet().to(device)
    checkpoint = torch.load(model_check)
    print(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    loss_function = nn.CrossEntropyLoss()
    model.eval()

    iteration = 0
    gods_accuracy = np.zeros(101)

    with torch.no_grad():
        for i, (instance, labels) in enumerate(valid_dl):

            instance = instance.to(device)
            labels = labels.to(device=device, dtype=torch.int64)

            log_probs = model(instance)
            loss = loss_function(log_probs, labels)
            pre_y = torch.max(log_probs, 1)[1]

            for j in range(0, 4):
                if pre_y[j] == labels[j]:
                    for p in range(0, 10):
                        for q in range(0, 101):
                            if instance[j][p][q] == 1:
                                gods_accuracy[q] += 1

    for j in range(0, gods_accuracy.size):
        if gods_accuracy[j] != 0:
            gods_accuracy[j] = '{:.3f}'.format((gods_accuracy[j] / gods_val[j]) * 100)
    with open('D:/model_param/test23/accuracy_valid_1.txt', 'a') as outFile:
                outFile.write(str(gods_accuracy))
    return gods_accuracy


best_model = 'D:/model_param/test23/itr_114000'
accuracy = accuracy_heroes(best_model)
print('Accuracy validation')
print(accuracy)
accuracy_ = np.sum(accuracy)/101
print(accuracy_)"""