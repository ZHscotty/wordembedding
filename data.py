import pickle
import os
import jieba
from collections import Counter


class Data:
    def __init__(self):
        self.path = 'E:\冰鉴\搜狐数据集'
        self.texts = self.load_text()[:100]
        self.stopword = self.load_stopword()
        self.tokens = self.tokenize(self.texts, self.stopword)
        self.dic = self.creat_dic(self.tokens)
        self.word2index = self.creat_word2index(self.dic)
        self.token_ids = self.get_tokenid(self.tokens, self.word2index)
        self.index2word = self.index2word()

    def load_text(self):
        with open(os.path.join(self.path, '文本列表.pkl'), 'rb') as f:
            texts = pickle.load(f)
        return texts

    def load_stopword(self):
        p = './stopword/哈工大停用词表.txt'
        with open(p, 'r', encoding='utf-8') as f:
            stopword = [x.strip() for x in f.readlines()]
        return stopword

    def tokenize(self, texts, stopword):
        texts_token = [jieba.lcut(x) for x in texts]
        tokens = []
        for x in texts_token:
            token = []
            temp = []
            for y in x:
                if len(y) == 1:
                    if 65296 <= ord(y) <= 65305:
                        temp.append(y)
                    elif y < '\u4e00' or y > '\u9fff':
                        pass
                    else:
                        if len(temp) != 0:
                            if len(temp) < 5:
                                token.append(''.join(temp))
                                temp = []
                            else:
                                temp = []
                        if y not in stopword:
                            token.append(y)
                else:
                    if y not in stopword:
                        token.append(y)
            if len(token) > 10:
                tokens.append(token)
        return tokens

    def creat_dic(self, tokens):
        temp = []
        dic = []
        for x in tokens:
            temp.extend(x)
        fre = dict(Counter(temp))
        fre = sorted(fre.items(), key=lambda x: x[1], reverse=True)

        for x in fre:
            dic.append(x[0])
        return dic

    def creat_word2index(self, dic):
        word2index = {x: index+1 for index, x in enumerate(dic)}
        word2index['UNK'] = 0
        return word2index

    def get_tokenid(self, tokens, word2index):
        token_ids = []
        for x in tokens:
            token_id = []
            for y in x:
                if y in word2index:
                    token_id.append(word2index[y])
                else:
                    token_id.append(word2index['UNK'])
            token_ids.append(token_id)
        return token_ids

    def index2word(self):
        index2word = {self.word2index[x]: x for x in self.word2index}
        return index2word