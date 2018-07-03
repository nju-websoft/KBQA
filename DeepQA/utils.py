# -*- coding:utf8 -*-

def get_ngram(text):
    """
    返回文本text 的n=3以内的token，返回结果以token从大到小排序
    :param text:  输入的文本，一般为实体name
    :return:
    """
    ngram = []  # token结果
    tokens = text.split()  # 以空格分开
    for i in range(len(tokens) + 1):
        for j in range(i):
            if i - j <= 3:  # tokens 长度最多为3
                temp = " ".join(tokens[j:i])
                if temp not in ngram:
                    ngram.append(temp)
    ngram = sorted(ngram, key=lambda x: len(x.split()), reverse=True)  # 将ngram中元素以长度进行排序
    return ngram

def www2fb(in_str):
    """
    将FB2M中的实体和属性的www.freebase.com/前缀去除
    :param in_str:
    :return:
    """
    if in_str.startswith("www.freebase.com"):
        in_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    if in_str == 'fb:m.07s9rl0':
        in_str = 'fb:m.02822'
    if in_str == 'fb:m.0bb56b6':
        in_str = 'fb:m.0dn0r'
    # Manual Correction
    if in_str == 'fb:m.01g81dw':
        in_str = 'fb:m.01g_bfh'
    if in_str == 'fb:m.0y7q89y':
        in_str = 'fb:m.0wrt1c5'
    if in_str == 'fb:m.0b0w7':
        in_str = 'fb:m.0fq0s89'
    if in_str == 'fb:m.09rmm6y':
        in_str = 'fb:m.03cnrcc'
    if in_str == 'fb:m.0crsn60':
        in_str = 'fb:m.02pnlqy'
    if in_str == 'fb:m.04t1f8y':
        in_str = 'fb:m.04t1fjr'
    if in_str == 'fb:m.027z990':
        in_str = 'fb:m.0ghdhcb'
    if in_str == 'fb:m.02xhc2v':
        in_str = 'fb:m.084sq'
    if in_str == 'fb:m.02z8b2h':
        in_str = 'fb:m.033vn1'
    if in_str == 'fb:m.0w43mcj':
        in_str = 'fb:m.0m0qffc'
    if in_str == 'fb:m.07rqy':
        in_str = 'fb:m.0py_0'
    if in_str == 'fb:m.0y9s5rm':
        in_str = 'fb:m.0ybxl2g'
    if in_str == 'fb:m.037ltr7':
        in_str = 'fb:m.0qjx99s'
    return "<"+in_str+">"

def pad_sentences(sentence, length, padding_index=0):
    """
    #将sentence(以词id)表示，padding到length长度
    :param sentence: 以词id表示的句子
    :param length: padding的长度
    :param padding_index: padding的字符，默认为0
    :return:
    """
    num_padding = length - len(sentence)  # 需要padding的长度
    if num_padding >0:
        new_sentence = sentence + [padding_index] * num_padding
    else:
        new_sentence = sentence[:length]  # 太长则进行截取
    return new_sentence




