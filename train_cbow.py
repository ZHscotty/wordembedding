from cbow import Model
from utils import prepare_data
from data import Data

d = Data()
train_data, train_label, dev_data, dev_label = prepare_data(window=5, d=d)
word2index = d.word2index
print('train_data shape:', train_data.shape)
print('train_label shape:', train_label.shape)
print('dev_data shape:', dev_data.shape)
print('dev_label shape:', dev_label.shape)

model = Model(word2index=word2index)


