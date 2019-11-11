from data import Data
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np


def prepare_data(window, d):
    tokens = d.token_ids
    word2index = d.word2index
    trainset, devset = train_test_split(tokens, test_size=0.3, random_state=40)
    train_data, train_label = create_data(trainset, window)
    train_data = np.array(train_data)
    # train_label = to_categorical(train_label, num_classes=len(word2index))
    dev_data, dev_label = create_data(devset, window)
    dev_data = np.array(dev_data)
    # dev_label = to_categorical(dev_label, num_classes=len(word2index))
    return train_data, train_label, dev_data, dev_label


def create_data(dataset, window):
    data = []
    label = []
    for x in dataset:
        start = 0
        while start <= len(x) - window:
            temp = x[start:start + window]
            label_example = []
            data_example = []
            for index in range(len(temp)):
                if index == window // 2:
                    label_example.append(temp[index])
                else:
                    data_example.append(temp[index])
            count = 0
            for m in temp:
                if m == 0:
                    count += 1
            if count < 2:
                data.append(data_example)
                label.append(label_example)
            start += 1
    return data, label


# if __name__ == '__main__':
#     d = Data()
#     index2word = d.index2word
#     print(index2word)
#     train_data, train_label, dev_data, devset = prepare_data(window=5, d=d)
#
#     for x in range(len(train_data)):
#         temp = []
#         for y in train_data[x]:
#             temp.append(index2word[y])
#         print(temp, index2word[train_label[x][0]])
