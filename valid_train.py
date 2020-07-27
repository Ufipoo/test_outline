import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader

print(torch.__version__)
from torch.nn.functional import softmax
import visdom
vis = visdom.Visdom()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#data = [(np.zeros(10), np.array([0, 1])), ('0 0 0 0 0 0 0 0 0 0'.split(), '01'), ('1 1 1 1 1 1 1 1 1 1'.split(), '10'), ('1 1 1 1 1 1 1 1 1 1'.split(), '10')]
#test_data = [('0 0 0 0 0 0 0 0 0 0'.split(), '01'), ('1 1 1 1 1 1 1 1 1 1'.split(), '10')]
win1 = np.array([0, 1])
win2 = np.array([1, 0])

info_match = []
file = open('upd_qonq.txt', 'r')
spisok = file.readlines()
for line in spisok:
    list_match = []
    parts = line.split()
    for i in parts:
        list_match.append(int(i))
    info_match.append(list_match)

data2 = []
for i in range(0, 10000):
    data3 = torch.Tensor([np.array(info_match[i][:10])])
    data2.append((data3, np.array(info_match[i][10:])))
#print(data2)


train_dl = DataLoader(data2[:8000], batch_size=4)
valid_dl = DataLoader(data2[8000:9000], batch_size=4)
tese_dl = DataLoader(data2[9000:], batch_size=4)


data1 = data2[:8000]
valid_data = data2[8000:9000]
test_data = data2[9000:10000]
print(test_data)
word_to_ix = {}
for sent, _ in data1 + test_data:
    for word in range(0, 10):
        if sent[0, word].item() not in word_to_ix:
            word_to_ix[sent[0, word].item()] = len(word_to_ix)

print('word to ix = ', word_to_ix)

#word_to_ix - элементы которые встречаются в выборках и их порядковый номер
#01 - победа 2 команды, 10 - победа 1 команды
size = 2
v_size = len(word_to_ix)
batch_size = 4
lr = 0.1


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(101, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


print('__________')


def make_bow_vector(sentence):
    vec = torch.zeros(101)
    for word in range(0, 10):
        vec[int(sentence[0, word])] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[str(label)]])

label_to_ix = {'[1 0]': 0, '[0 1]': 1}
model = NeuralNet()
model.to(device)
print('label to ix = ', label_to_ix)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


plot_acc = vis.line(Y=np.array([0]), X=np.array([0]))
plot_loss = vis.line(Y=np.array([0]), X=np.array([0]))
for i, (instance, label) in enumerate(train_dl):
    instance = instance.to(device)
    print(instance)
    print(i)
for epoch in range(100):
    IRL_target, NET_target = [], []
    m_loss = float(0)
    for i, (instance, label) in enumerate(train_dl):
        for ins, l in (instance, lebel)
        print(instance)

        model.zero_grad()
        bow_vec = make_bow_vector(instance)
        target = make_target(label, label_to_ix)
        log_probs = model(bow_vec)
        print(log_probs)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        m_loss += loss
        pre_y = torch.max(log_probs, 1)[1]
        print(int(pre_y))
        NET_target.append(np.int(pre_y))
        IRL_target.append(np.int(target))

    if (i + 1) % 500 == 0:
        print(i)
        m_loss2 = m_loss / float(i + 1)
        print('LOSS = ' + str(m_loss2))
        NET_target2 = np.array(NET_target)
        IRL_target2 = np.array(IRL_target)
        toch = float((IRL_target2 == NET_target2).astype(int).sum()) / float(IRL_target2.size)
        print('ACC = ' + str(toch))
        vis.line(Y=np.array([m_loss2.item()]), X=np.array([4 * epoch + (i / 500)]), win=plot_loss, update='append')
        vis.line(Y=np.array([toch]), X=np.array([4 * epoch + (i / 500)]), win=plot_acc, update='append')




'''


print('_______________________________________' + '\n' + 'TEST')
NET_test, IRL_test = [], []
with torch.no_grad():

    for instance, label in test_data:
        bow_vec = make_bow_vector(instance)
        print(bow_vec)
        log_probs = model(bow_vec)
        print(log_probs)
        # m = nn.Softmax()
        outputs = softmax(log_probs, dim=1)

        win_team_index = torch.argmax(outputs).item()
        if win_team_index == 0:
            print('win team - 1')
        else:
            print('win team - 2')
        print('log_probs = ', log_probs)
        print('outputs = ', outputs)

        pred_y = torch.max(log_probs, 1)[1]
        NET_test.append(np.int(pred_y))
        IRL_test.append(np.int(target))
NET_test = np.array(NET_test)
IRL_test = np.array(IRL_test)
toch_test = float((IRL_test == NET_test).astype(int).sum()) / float(IRL_test.size)
print(toch_test)

# # m = nn.Softmax()
# #слева вероятность победы 1 команды, справа 2 команды. Одно из значение больше другого, значит оно ближе к правде(по модели)
# print(next(model.parameters())[:, word_to_ix[1]])
# answer = next(model.parameters())[:, word_to_ix[1]]
# print(softmax(answer, dim = 0))



'''




