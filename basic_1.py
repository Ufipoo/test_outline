import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
print(torch.__version__)

#data = [(np.zeros(10), np.array([0, 1])), ('0 0 0 0 0 0 0 0 0 0'.split(), '01'), ('1 1 1 1 1 1 1 1 1 1'.split(), '10'), ('1 1 1 1 1 1 1 1 1 1'.split(), '10')]
#test_data = [('0 0 0 0 0 0 0 0 0 0'.split(), '01'), ('1 1 1 1 1 1 1 1 1 1'.split(), '10')]
win1 = np.array([0, 1])
win2 = np.array([1, 0])

data = [(torch.zeros(1, 10), np.array([0, 1])), (torch.zeros(1, 10), np.array([0, 1])), (torch.ones(1, 10), np.array([1, 0])), (torch.ones(1, 10), np.array([1, 0]))]
test_data = [(torch.zeros(1, 10), np.array([0, 1])), (torch.ones(1, 10), np.array([1, 0]))]


word_to_ix = {}
for sent, _ in data + test_data:
    for word in range(0, 10):
        if sent[0, word].item() not in word_to_ix:
            word_to_ix[sent[0, word].item()] = len(word_to_ix)

print('word to ix = ', word_to_ix)

#word_to_ix - элементы которые встречаются в выборках и их порядковый номер
#01 - победа 2 команды, 10 - победа 1 команды
size = 2
v_size = len(word_to_ix)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in range(0, 10):
        vec[word_to_ix[sentence[0, word].item()]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[str(label)]])

label_to_ix = {'[1 0]': 0, '[0 1]': 1}
model = nn.Linear(v_size, size)
print('label to ix = ', label_to_ix)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for instance, label in data:
        model.zero_grad()
        print(instance)

        bow_vec = make_bow_vector(instance, word_to_ix)
        target = make_target(label, label_to_ix)

        log_probs = model(bow_vec)

        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
from  torch.nn.functional import softmax
with torch.no_grad():
    for instance, label in test_data:
        print(instance)
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = model(bow_vec)
        # m = nn.Softmax()
        print(log_probs.shape)
        outputs = softmax(log_probs, dim = 1)

        win_team_index = torch.argmax(outputs).item()
        if win_team_index == 0:
            print('win team - 1')
        else:
            print('win team - 2')
        print('log_probs = ', log_probs)
        print('outputs = ', outputs)

# # m = nn.Softmax()
# #слева вероятность победы 1 команды, справа 2 команды. Одно из значение больше другого, значит оно ближе к правде(по модели)
# print(next(model.parameters())[:, word_to_ix[1]])
# answer = next(model.parameters())[:, word_to_ix[1]]
# print(softmax(answer, dim = 0))
