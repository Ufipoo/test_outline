from sklearn import datasets
import numpy as np
import random

import matplotlib.pyplot as plt

#Загрузка датасета
digits = datasets.load_digits()


#Показать случайные картинки
'''
fig, axes = plt.subplots(4,4)

axes=axes.flatten()
plt.figure()
plt.imshow(digits.images[0])
#plt.show()
plt.imshow(digits.images[1])
plt.colorbar()
plt.grid(False)
plt.show()'''

"""
for i, ax in enumerate(axes):
    dig_ind=np.random.randint(0,len(digits.images))
    ax.imshow(digits.images[dig_ind].reshape(8,8))
    ax.set_title(digits.target[dig_ind])
plt.show()
"""

#Посчитать картинок какого класса сколько
dic={x:0 for x in range(10)}
for dig in digits.target:
    dic[dig]+=1
print(dic)


def prepare_data(data, avg):
    """
    Подготавливает данные для кореляционного классификатора
    :param data: np.array, данные (размер выборки, количество пикселей
    :return: data: np.array, данные (размер выборки, количество пикселей
    """
    return data - avg


def train_val_test_split(data, labels):

    """
    Делит выборку на обучающий и тестовый датасет
    :param data: np.array, данные (размер выборки, количество пикселей)
    :param labels: np.array, метки (размер выборки,)
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    size_train = (data.shape[0]//100)*80
    if data.shape[0]%2 == 0:
        size_v_t = int((data.shape[0] - size_train)/2)

    else:
        size_train += 1
        size_v_t = int((data.shape[0] - size_train)/2)
    train_data = data[:size_train]
    train_labels = labels[:size_train]
    validation_data = data[size_train:size_train + size_v_t]
    validation_labels = labels[size_train:size_train + size_v_t]
    test_data = data[size_train + size_v_t:]
    test_labels = labels[size_train + size_v_t:]
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def softmax(vec):
    vec = np.array(vec)
    max_e = np.max(vec)
    temp = []

    for i in range(0, 10):
        temp.append(np.exp(vec[i] - max_e)/np.sum(np.exp(vec - max_e)))
        #vec[i] = np.exp(vec[i]-max_e)/np.sum(np.exp(vec-max_e))
    return temp





class CorelationClassifier:

    def __init__(self, classes_count=10):
        self.classes_count=classes_count
        self.averages = []



    def fit(self, data, labels):
        """
        Производит обучение алгоритма на заданном датасете
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return: СКЛАДЫВАЕМ КАРТИНКИ ОДНОГО КЛАССА И ДЕЛИМ НА КОЛИЧЕСТВО ИХ
        ВЕКТОР ИЗ СРЕДНИХ ВЕКТОРОВ
        """

        zero_v = []
        for i in range(0, 64):
            zero_v.append(0)
        for i in range(0, 10):
            sum = 0
            k = 0
            for j in range(0, len(data)):
                if labels[j] == i:
                    sum += np.array(data[j])
                    k += 1

            if k == 0:
                self.averages.append(zero_v)
            else:
                mn_sum = sum / k
                self.averages.append(mn_sum)
        self.averages = np.array(self.averages)
        return self.averages




    def predict(self, data):
        """
        Предсказывает вектор вероятностей для каждого наблюдения в выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :return: np.array, результаты (len(data), count_of_classes)
        ВЕКТОР ВЕРОЯТНОСТЕЙ ДЛЯ КАЖДОГО ЭЛЕМЕНТА В ВЫБОРКИ
        """
        count_of_classes = []
        for i in range(0, len(data)):
            t = []
            for j in range(0, 10):
                t.append(np.dot(data[i], self.averages[j].T))
            count_of_classes.append(softmax(t))
            #print(count_of_classes)
        self.classes_count = np.array(count_of_classes)
        return self.classes_count



    def accuracy(self, data, labels):
        """
        Оценивает точность (accuracy) алгоритма по выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return: ТОЧНОСТЬ
        """
        acc = 0
        max_vec = []
        for i in range(0, len(data)):
            max_vec.append(np.argmax(self.classes_count[i]))
        for i in range(0, len(data)):
            if max_vec[i] == labels[i]:
                acc += 1
        return acc/len(data)




train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_val_test_split(digits.data, digits.target)

train_data = prepare_data(train_data, 8)
validation_data = prepare_data(validation_data, 8)
test_data = prepare_data(test_data, 8)


#Посчитать картинок какого класса сколько в обучающем датасете
dic={x:0 for x in range(10)}
for dig in train_labels:
    dic[dig]+=1
print(dic)


classifier = CorelationClassifier()
print(classifier.fit(train_data, train_labels))
print(classifier.predict(train_data))
print('Training accuracy ' + str(classifier.accuracy(train_data, train_labels)))
classifier.predict(validation_data)
print('Validation accuracy ' + str(classifier.accuracy(validation_data, validation_labels)))


target = []
for i in range(0, len(validation_data)):
    target.append(np.argmax(classifier.predict(validation_data)[i]))
recall_precision = []
for i in range(0, 10):
    TP_FN_FP = [0, 0, 0]
    for j in range(0, len(validation_data)):
        if target[j] == i:
            if i == validation_labels[j]:
                TP_FN_FP[0] += 1
            elif i != validation_labels[j]:
                TP_FN_FP[1] += 1
        elif target[j] != i:
            if i == validation_labels[j]:
                TP_FN_FP[2] += 1
    recall_precision.append(TP_FN_FP)
print(recall_precision)

