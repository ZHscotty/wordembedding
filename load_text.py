# sougou dataset 1411996 samples
import pickle
path = 'E:\冰鉴\搜狐数据集\\news_sohusite_xml.txt'
text = []
with open(path, 'r', encoding='utf-8') as f:
    while True:
        str = ''
        for x in range(6):
            str += f.readline()
        if str == '':
            break
        text.append(str)

with open('E:\冰鉴\搜狐数据集\文本列表.pkl', 'wb') as f:
    pickle.dump(text, f)