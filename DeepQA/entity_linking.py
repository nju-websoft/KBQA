# -*- coding:utf8 -*-
import pickle
from string import punctuation
from fuzzywuzzy import fuzz  # 字符串模糊匹配的工具
from nltk.corpus import stopwords
from utils import get_ngram

inverted_index = {}
stopword = set(stopwords.words('english'))  # 获取英文的停用词，如a,the等功能性单词

def get_stat_inverted_index(filename):
    """
     获取倒排文档中，对应实体数目最多的token
    :param filename: 索引文件路径
    :return:
    """
    with open(filename, "rb") as handler:
        global inverted_index
        inverted_index = pickle.load(handler)
    print("Total type of text: {}".format(len(inverted_index)))
    max_len = 0
    _entry = ""
    for entry, value in inverted_index.items():
        if len(value) > max_len:
            max_len = len(value)
            _entry = entry
    print("Max Length of entry is {}, text is {}".format(max_len, str(_entry)))

def entity_linking(pred_mention, top_num):
    """
    对问题进行实体链接
    :param pred_mention:  问题
    :param top_num: 返回实体个数
    :return:
    """
    C = []
    C_scored = []
    tokens = get_ngram(pred_mention)

    have_find = []
    for item in tokens:
        if item in stopword:  # 省略停用词
            continue
        if item in punctuation:  # 省略标点符号
            continue
        t = 0
        for tok in have_find:  # 不查找子token
            token = tok.split()
            token1 = item.split()
            if set(token1).issubset(set(token)):
                t = 1
                break
        if t == 1:
            continue
        if item in inverted_index.keys():
            find = []
            for a in inverted_index[item]:
                a.append(item)
                find.append(a)
            C.extend(find)  # 查找包含该token的对应的实体

            have_find.append(item)

    newC = []
    for a in C:
        if a not in newC:
            newC.append(a)
    C = newC

    for mid_text_type in C:
        score = fuzz.ratio(mid_text_type[1], mid_text_type[3]) / 100.0  # 基于模糊匹配对找到的实体进行排序，基于该实体名字和问题之间相似度
        C_scored.append((mid_text_type, score))

    C_scored.sort(key=lambda t: t[1], reverse=True)  # 以分数进行倒排
    cand_mids = C_scored[:top_num]
    return cand_mids  #[[mid,name,type,mention].score]

if __name__ == "__main__":
    input_mentions = "what is the book e about"
    get_stat_inverted_index("./data/inverted_index.pkl")
    entity_linking(input_mentions, 20)
