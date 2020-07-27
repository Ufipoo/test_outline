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


label_to_ix = {'[1 0]': 0, '[0 1]': 1}

def make_vector(sentence):
    vec_hero = []
    for i in range(0, 10):
        hero = np.zeros(101)
        hero[sentence[i]] = 1
        vec_hero.append(hero)
    return vec_hero

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[str(label)]])

batch_size = 4


def make_vec(inst):
    temp = torch.Tensor(10,101)
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
        self.fc1 = nn.Linear(101, 1024)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = sum_conc(out)
        out = out.to(device)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

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

gods_test = np.zeros(101)
for i in range(9000, 10000):
    for word in range(0, 10):
        gods_test[info_match2[i][word]] += 1
print(gods_test)


data4 = []
data5 = []
for i in range(8000, 10000):
    instance = make_vector(info_match2[i][:10])
    target = make_target(np.array(info_match2[i][10:]), label_to_ix)
    data5.append(target)
    data4.append(instance)


data4 = torch.Tensor(data4)
data5 = torch.Tensor(data5)

x_valid = data4[:1000]
y_valid = data5[:1000]
x_test = data4[1000:2000]
y_test = data5[1000:2000]

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

test_ds = TensorDataset(x_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size)


def accuracy_heroes(model_check, samples):
    model = NeuralNet().to(device)
    checkpoint = torch.load(model_check)
    model.load_state_dict(checkpoint['model_state_dict'])

    loss_function = nn.CrossEntropyLoss()
    model.eval()

    iteration = 0
    gods_accuracy = np.zeros(101)
    NET_target, IRL_target = [], []
    if samples == 'validation':
        dl = valid_dl
    elif samples == 'test':
        dl = test_dl

    with torch.no_grad():
        for i, (instance, labels) in enumerate(dl):

            instance = instance.to(device)
            labels = labels.to(device=device, dtype=torch.int64)

            log_probs = model(instance)
            pre_y = torch.max(log_probs, 1)[1]

            NET_target.append(pre_y.cpu().data.numpy())
            IRL_target.append(labels.cpu().data.numpy())

            for j in range(0, 4):
                if pre_y[j] == labels[j]:
                    for p in range(0, 10):
                        for q in range(0, 101):
                            if instance[j][p][q] == 1:
                                gods_accuracy[q] += 1

    NET_target2 = np.array(NET_target)
    IRL_target2 = np.array(IRL_target)
    accua = float((IRL_target2 == NET_target2).astype(int).sum()) / float(IRL_target2.size)
    print('ACCURACY')
    print(accua * 100)

    if samples == 'validation':
        for j in range(0, gods_accuracy.size):
            if gods_accuracy[j] != 0:
                gods_accuracy[j] = '{:.3f}'.format((gods_accuracy[j] / gods_val[j]) * 100)

        for j in range(0, gods_accuracy.size):
            with open('D:/model_param/test_end_results/2_model_accuracy_valid2.txt', 'a') as outFile:
                outFile.write(str(gods_accuracy[j]) + '\n')

        with open('D:/model_param/test_end_results/2_model_accuracy_valid2.txt', 'a') as outFile:
            outFile.write('_____________________________________' + '\n' + 'Accuracy = ' + str(
                accua * 100) + '\n' + 'Accuracy heroes = ' + str(np.sum(gods_accuracy) / 101))

    elif samples == 'test':
        for j in range(0, gods_accuracy.size):
            if gods_accuracy[j] != 0:
                gods_accuracy[j] = '{:.3f}'.format((gods_accuracy[j] / gods_test[j]) * 100)

        for j in range(0, gods_accuracy.size):
            with open('D:/model_param/test_end_results/2_model_accuracy_test2.txt', 'a') as outFile:
                outFile.write(str(gods_accuracy[j]) + '\n')

        with open('D:/model_param/test_end_results/2_model_accuracy_test2.txt', 'a') as outFile:
            outFile.write('_____________________________________' + '\n' + 'Accuracy = ' + str(
                accua * 100) + '\n' + 'Accuracy heroes = ' + str(np.sum(gods_accuracy) / 101))

    return gods_accuracy

best_model = 'D:/model_param/test23/itr_114000'
accuracy = accuracy_heroes(best_model, 'validation')
print('Accuracy validation')
print(accuracy)
accuracy_ = np.sum(accuracy)/101
print(accuracy_)

accuracy = accuracy_heroes(best_model, 'test')
print('Accuracy test')
print(accuracy)
accuracy_ = np.sum(accuracy)/101
print(accuracy_)