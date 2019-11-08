# sougou dataset 1411996 samples
import pickle
import re
from bs4 import BeautifulSoup
path = 'E:\冰鉴\搜狐数据集\\news_sohusite_xml.txt'
text = []
with open(path, 'r', encoding='utf-8') as f:
    while True:
        str = ''
        for x in range(6):
            str += f.readline()
        if str == '':
            break
        pattern = '<content>(.*?)</content>'
        r = re.findall(pattern, str)[0]
        text.append(r)

with open('E:\冰鉴\搜狐数据集\文本列表.pkl', 'wb') as f:
    pickle.dump(text, f)