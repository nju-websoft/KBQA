# -*- coding:utf8 -*-
from nltk.tokenize import word_tokenize
import re
import json
import random
from nltk.corpus import stopwords
from string import punctuation
import data_loader
import pickle
from utils import *
from entity_linking import *
from config import config

def producevob():
    """
    获取训练集、验证集、测试集中出现过的所有的词语，构成本次使用的字典
    本次未使用
    :return:
    """
    with open("./data/freebase/fb5m_ent_name.pkl", "rb") as f:  # 加载实体-名字字典
        fb5m_ent_name = dict(pickle.load(f))
    ignore = 0
    id = 0
    vocab = {}  # 词表
    paths = ["./data/SimpleQuestion/annotated_fb_data_train.txt","./data/SimpleQuestion/annotated_fb_data_text.txt","./data/SimpleQuestion/annotated_fb_data_valid.txt"]
    for path in paths:
        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                fields = line.strip().split('\t')  # 将每行数据用\t隔开
                sub = www2fb(fields[0])
                rel = fields[1].split('www.freebase.com/')[-1].replace('/', ' ')
                obj = www2fb(fields[2])  # 获取实体
                question = fields[3]  # 获取问题
                if obj not in fb5m_ent_name.keys():  # 如果该实体没有name
                    ignore += 1
                    continue
                if sub not in fb5m_ent_name.keys():
                    ignore += 1
                    continue

                question = question.replace('\\\\', ' ')  # 将question中\\除去
                question = re.sub("[-_{}/.]", " ", question).lower()
                tokens = word_tokenize(question)  # 将问题进行分词
                for i in range(len(tokens)):
                    if tokens[i] not in vocab.keys():
                        vocab[tokens[i]] = id
                        id += 1
                sub = fb5m_ent_name[sub].replace('\\\\', ' ')
                sub = re.sub("[-_{}/.]", " ", sub).lower()
                tokens = word_tokenize(sub)
                for i in range(len(tokens)):
                    if tokens[i] not in vocab.keys():
                        vocab[tokens[i]] = id
                        id += 1
                rel = fb5m_ent_name[rel].replace('\\\\', ' ')
                rel = re.sub("[-_{}/.]", " ", rel).lower()
                tokens = word_tokenize(rel)  # 将问题进行分词
                for i in range(len(tokens)):
                    if tokens[i] not in vocab.keys():
                        vocab[tokens[i]] = id
                        id += 1
                obj = fb5m_ent_name[obj].replace('\\\\', ' ')
                obj = re.sub("[-_{}/.]", " ", obj).lower()
                tokens = word_tokenize(obj)
                for i in range(len(tokens)):
                    if tokens[i] not in vocab.keys():
                        vocab[tokens[i]] = id
                        id += 1

    print("一共出现了",id, "个词")

    # pickle序列化
    with open('./data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, 2)

def produce_fb5m(filepath="./data/freebase/FB5M.name.txt"):
    """
    根据FB5M文件，抽取出每一个名字对应的name,name为分词
    :param filepath: 文件路径
    :return: fb5m_ent_name 的一个字典
    """
    fb5m_ent_name = {}  # 根据实体获取相应的名字
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            fields = line.strip().split('\t')  # 将每行数据用\t隔开
            entity = fields[0]  # 获取实体
            rel = fields[1]
            name = fields[2]  # 获取实体名字
            if rel != '<fb:type.object.name>':  # 简单起见，每一个实体只保留name属性
                continue
            if name[0] == '"':
                name = name[1:]
            if name[-1] == '"':
                name = name[:-1]
            fb5m_ent_name[entity] = name  # 默认每一个实体只有一个name
    # pickle序列化
    with open('./data/freebase/fb5m_ent_name.pkl', 'wb') as f:
        pickle.dump(fb5m_ent_name, f, 2)
    return fb5m_ent_name

def word_index(vocab, question):
    """
    问句、实体、属性进行分词以及预处理模块，分词，建立索引，返回的是一个list,其每一个元素指向一个word
    :param vocab: 词汇表
    :param question: 问句
    :return:
    """
    question = question.replace('\\\\', ' ')  # 将question中\\除去
    question = re.sub("[-_{}/.]", " ", question).lower()
    tokens = word_tokenize(question)  # 将问题进行分词
    for i in range(len(tokens)):
        if tokens[i] not in vocab.keys():
            # print("找不到该词" + tokens[i])
            tokens[i] = '<unk>'  # 对于找不到的词规到‘unk’
    question = [vocab[t] for t in tokens]
    return question

def produce_data(filepath="./data/SimpleQuestion/annotated_fb_data_test.txt", savepath = "./data/SimpleQuestion/testdata.pkl"):
    """
    对训练集、测试集、验证集文件进行处理，将实体转换成实体name， 将name、question进行分词处理，并转换成词id
    :param filepath: 训练集、测试集、验证集所在的文件路径
    :param savepath: 文件保存路径
    :return:[[question, sub, rel, obj],] 列表
    """
    vocab, embd = data_loader.load_glove()
    with open("./data/freebase/fb5m_ent_name.pkl", "rb") as f:  # 加载实体-名字字典
        fb5m_ent_name = dict(pickle.load(f))
    ignore = 0
    question_sample = []
    question_pure = []
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            fields = line.strip().split('\t')  # 将每行数据用\t隔开
            sub = www2fb(fields[0])
            rel = fields[1].split('www.freebase.com/')[-1].replace('/', ' ')
            obj = www2fb(fields[2])  # 获取实体
            question = fields[3]  # 获取问题
            if obj not in fb5m_ent_name.keys():  # 如果该实体没有name
                ignore += 1
                continue
            if sub not in fb5m_ent_name.keys():
                ignore += 1
                continue
            sub = fb5m_ent_name[sub]
            obj = fb5m_ent_name[obj]
            question_pure.append([question, sub, rel, obj])

            question = word_index(vocab, question)  # 将问题索引化
            sub = word_index(vocab, sub)  # 分词建索引
            obj = word_index(vocab, obj)
            rel = word_index(vocab, rel)
            question_sample.append([question, sub, rel, obj])
    print("共有" + str(ignore) + "个问题的答案没有名字")
    with open(savepath, "wb") as f:
        pickle.dump(question_sample, f, 2)
    with open("./data/SimpleQuestion/testdata_pure.pkl", "wb") as f:
        pickle.dump(question_pure, f, 2)

def produce_entity_fb2m():
    """
    处理FB2M，进行切词、词ID化
    :return:
    """
    vocab, embd = data_loader.load_glove()
    with open("./data/freebase/fb5m_ent_name.pkl", "rb") as f:  # 加载实体-名字字典  blue christmas
        fb5m_ent_name = dict(pickle.load(f))
    print("FB5M中实体数目为" + str(len(fb5m_ent_name.keys()))) # 3972329
    fb2m_sub_obj = {}   # 存放fb2m中主语实体和宾语（多个）实体 key=subname value=[(rel,objname)]
    fb2m_obj_sub = {}
    fb2m_index = []
    fb2m = []
    with open("./data/freebase/freebase-FB2M.txt", 'r', encoding='utf-8') as file:
        for line in file.readlines():
            fields = line.strip().split('\t')  # 将每行数据用\t隔开
            sub = www2fb(fields[0])  # 获取实体 <fb:m.0f8v12b>
            if sub not in fb5m_ent_name.keys():
                continue
            rel = fields[1].split('www.freebase.com/')[-1].replace('/', ' ')
            obj = www2fb(fields[2])
            objs = obj.split(" ")  # 有多个宾语
            objss = []
            for obj in objs:
                if obj in fb5m_ent_name.keys():  # 该实体拥有name
                    objss.append(fb5m_ent_name[obj])
            if len(objss) <= 0:
                continue
            sub = fb5m_ent_name[sub]  # 转换为词

            right = []
            for obj in objss:
                right.append((rel,obj))

            if sub not in fb2m_sub_obj.keys():
                fb2m_sub_obj[sub] = right  # 保存左实体关系
            else:
                fb2m_sub_obj[sub].extend(right)

            for obj in objss:
                if obj not in fb2m_obj_sub.keys():
                    fb2m_obj_sub[obj] = [(rel, sub)]
                else:
                    fb2m_obj_sub[obj].append((rel, sub))
                fb2m.append((sub,rel,obj))
                fb2m_index.append((word_index(vocab, sub),word_index(vocab, rel),word_index(vocab, obj)))  #将索引化后的FB2M进行保存，便于产生反例

    with open("./data/fb2m_sub_obj.pkl","wb") as f:  #
        pickle.dump(fb2m_sub_obj, f, 2)

    with open("./data/fb2m_obj_sub.pkl","wb") as f:
        pickle.dump(fb2m_obj_sub, f, 2)

    with open("./data/fb2m_index.pkl","wb") as f:
        pickle.dump(fb2m_index, f, 2)

    with open("./data/fb2m.pkl","wb") as f:
        pickle.dump(fb2m, f, 2)
    return

def clear_fb2m():
    """
    生成的fb2m_index 中存在大量重复的三元组，进行去重
    :return:
    """
    with open("./data/freebase/fb2m_index.pkl", "rb") as f:  # 加名字index字典
         fb2m_index = list(pickle.load(f))
    fb2m = []
    i = 0
    before = None
    for item in fb2m_index:
        if i == 0:
            before = item
            fb2m.append(item)
        else:
            if item != before:
                fb2m.append(item)
                before = item
        i+=1
        print(i)

    with open("./data/fb2m_index.pkl","wb") as f:
        pickle.dump(fb2m, f, 2)

def random_entity(fb2m, size, list):
    """
    #随即中fb2m中产生三元组
    :param fb2m:
    :param size:
    :param list:
    :return:
    """
    while True:
        if len(list) >= size:
            break
        index = random.randint(0, len(fb2m) - 1)
        if fb2m[index] not in list:
            list.append(fb2m[index])
    return list

def prodece_candidate_bin_entity_top80():
    """
    根据生成的候选实体,将包含该实体的相关前80个三元组作为候选三元组
    :return:
    """
    with open("./data/candidate_entity_top40_test.pkl", "rb") as f:
        candidate_entity_top40 = dict(pickle.load(f))  # key=question, value=[[mid,name,type,mention],score]
    with open("./data/freebase/fb2m_sub_obj.pkl","rb") as f:  # key=mid value=[(rel,mid)]
        fb2m_sub_obj = dict(pickle.load(f))
    with open("./data/freebase/fb2m_obj_sub.pkl","rb") as f:  # key=mid value=[(rel,mid)]
        fb2m_obj_sub = dict(pickle.load(f))
    with open("./data/freebase/fb2m.pkl","rb") as f:  # [(sub,rel,obj)]
        fb2m = list(pickle.load(f))

    candidate_bin_triplet_top80 = {}  # 为每一个问题生成80个候选宾语实体的name

    less80 = 0
    sum = 0
    ignore = 0

    #遍历每一个问题
    for question,value in candidate_entity_top40.items():
        sum += 1
        print(sum)
        for a in value:
            submid = a[0][1]  # 获取实体name
            if submid in fb2m_sub_obj.keys():  # 有三元组将该实体作为主语
                for right in fb2m_sub_obj[submid]:
                    if question not in candidate_bin_triplet_top80.keys():
                        candidate_bin_triplet_top80[question] = [(submid, right[0], right[1])]
                    else:
                        if (submid, right[0], right[1]) not in candidate_bin_triplet_top80[question] and (len(candidate_bin_triplet_top80[question])<80):  # 对于同一个question,三元组不重复
                            candidate_bin_triplet_top80[question].append((submid, right[0], right[1]))
            if submid in fb2m_obj_sub.keys():  # 有三元组将该实体作为宾语
                for left in fb2m_obj_sub[submid]:
                    if question not in candidate_bin_triplet_top80.keys():
                        candidate_bin_triplet_top80[question] = [(left[1], left[0], submid)]
                    else:
                        if (left[1], left[0], submid) not in candidate_bin_triplet_top80[question] and (len(candidate_bin_triplet_top80[question])<80):  # 对于同一个question,三元组不重复
                            candidate_bin_triplet_top80[question].append((left[1], left[0], submid))

        if question in candidate_bin_triplet_top80.keys():
            a = candidate_bin_triplet_top80[question]
            if len(a) < 80:  # 不足的随机补齐
                less80 += 1
                random_entity(fb2m, 80 , a)
                print(len(a))
        else:
            a = []
            ignore += 1
            neg = random_entity(fb2m, 80 , a)
            candidate_bin_triplet_top80[question] = neg
    print("忽略问题个数" + str(ignore)) # 忽略问题个数147
    print("不足80的"+str(less80))  # 不足80的2592
    print("问题总数"+str(sum))  # 问题总数10804

    with open('./data/candidate_bin_triplet_top80_test.pkl', 'wb') as f:
        pickle.dump(candidate_bin_triplet_top80, f, 2)  # key=question value =[name,name]

def produce_valid_sample():
    """
    # 产生验证数据集,进行词索引化
    :return:
    """
    with open('./data/candidate_bin_triplet_top80_test.pkl', 'rb') as f:
        candidate_bin_triplet_top80_valid = dict(pickle.load(f))  # key=question value =[(name,rel,name)]
    vocab, embd = data_loader.load_glove()
    with open('./data/SimpleQuestion/testdata_pure.pkl', 'rb') as f:
        validdata = list(pickle.load(f))  # [[question, sub, rel, obj], ]  # 未分词

    question_list = []  # 问题列表  len
    can_tripelt_list = []  # 候选triplet列表  len*80
    answer_index = []  # 正确答案在候选列表中的index
    for line in validdata:
        question = line[0]
        answer = (line[1],line[2],line[3])
        if question not in candidate_bin_triplet_top80_valid.keys():
            continue
        for index,can in enumerate(candidate_bin_triplet_top80_valid[question]):
            if can == answer:  # 只记录在候选实体中有正确答案的问题
                question_list.append(question)
                can_tripelt_list.append(candidate_bin_triplet_top80_valid[question])
                answer_index.append(index)
                break

            if index == len(candidate_bin_triplet_top80_valid[question]) - 1: # 候选集中没有答案
                ii = random.randint(0,len(candidate_bin_triplet_top80_valid[question])-1)  # 随机将正确答案插入
                candidate_bin_triplet_top80_valid[question][ii] = answer
                question_list.append(question)
                can_tripelt_list.append(candidate_bin_triplet_top80_valid[question])
                answer_index.append(ii)
                break
    # 验证集的长度
    len_samples = len(question_list)  # # 问题总数8514
    print("验证集长度为"+str(len_samples))

    # 对验证集中所有数据进行分词，以及词索引化
    questions = [word_index(vocab, question) for question in question_list]  # len*tarinsize
    question_list.clear()
    tripelt = []
    for item in can_tripelt_list:
        b = [word_index(vocab, a[0])+word_index(vocab, a[1])+word_index(vocab, a[2]) for a in item]  # 将三元组连接起来当作一个句子
        tripelt.append(b)
    can_tripelt_list.clear()

    question_len = [len(s) for s in questions]
    can_triplet_len = []  # 一个问题有多个候选实体，因此是一个二位矩阵 len * 80
    for a in tripelt:
        a_len = []
        for s in a:
            if len(s) > config.num_step:
                a_len.append(config.num_step)
            else:
                a_len.append(len(s))
        can_triplet_len.append(a_len)

    # 对数据进行padding操作，使得等长
    ques_pad = [pad_sentences(item, config.num_step) for item in questions]
    questions.clear()

    ans_pad = []
    for neg in tripelt:
        result1 = [pad_sentences(item, config.num_step) for item in neg]
        ans_pad.append(result1)
    tripelt.clear()
    print("数据padding成功！")

    dataset = [ques_pad, ans_pad, answer_index, question_len, can_triplet_len]
    print(str(ques_pad[0]))
    print(str(ans_pad[0]))
    print(str(answer_index[0]))
    print(str(question_len[0]))
    print(str(can_triplet_len[0]))
    with open("./data/test_sample_all.pkl",'wb') as f:
        pickle.dump(dataset, f, 2)

    return dataset

if __name__=='__main__':
    produce_valid_sample()

def produce_inverted_index():
    """
    #产生实体链接时的倒排索引字典 key=token value=[[mid,name,type],...]
    :return:
    """
    stopword = set(stopwords.words('english'))  # 获取英文的停用词，如a,the等功能性单词
    inverted_index = {}
    with open("./data/freebase/FB5M.name.txt","r",encoding="utf-8") as f:
        for line in f.readlines():
            fields = line.strip().split('\t')  # 将每行数据用\t隔开
            entity = fields[0]  # 获取实体
            nametype = fields[1]  # name的类型，一般只有两种type.object.name，common.topic.alias
            name = fields[2]  # 获取实体名字
            if name[0] == '"':
                name = name[1:]
            if name[-1] == '"':
                name = name[:-1]
            tokens = get_ngram(name)
            for token in tokens:
                if token in stopword:  # 省略停用词
                    continue
                if token in punctuation:  # 省略标点符号
                    continue
                if token not in inverted_index.keys():
                    t = []
                else:
                    t = inverted_index[token]
                t.append([entity, name, nametype])
                inverted_index[token] = t
    # pickle序列化
    with open('./data/inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f, 2)
    with open('./data/inverted_index.text', 'w') as f:
        f.write(json.dumps(inverted_index))

def produce_candidate_entity_top40():
    """
    #产生候选实体，最多40个 key=question value=[([mid,name,type,mention],score)....]
    :return:
    """
    get_stat_inverted_index("./data/inverted_index.pkl")  # 加载索引文件
    candidate_entity_top40 = {}
    with open("./data/SimpleQuestion/annotated_fb_data_test.txt","r",encoding="UTF-8") as f:
        i = 0
        for line in f.readlines():
            print(i)
            i += 1
            fields = line.strip().split('\t')  # 将每行数据用\t隔开
            sub = www2fb(fields[0])  # 获取实体
            pred = www2fb(fields[1])  # 获取属性
            obj = www2fb(fields[2])  # 获取属性值
            question = fields[-1]  # 获取问题
            cand_mids = entity_linking(question, 40)  # [[mid,name,type,mention],score]
            candidate_entity_top40[question] = cand_mids
        # pickle序列化
    with open('./data/candidate_entity_top40_test.pkl', 'wb') as f:
        pickle.dump(candidate_entity_top40, f, 2)
