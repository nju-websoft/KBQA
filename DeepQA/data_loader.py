from config import config
import numpy as np
import random
import pickle
#将词表以字典形式返回，加快搜索速度
def load_glove(filename="./data/glove/glove.6B.50d.txt", emb_size=50):
    """
    加载预训练的词向量文件
    :param filename:  词向量路径
    :param emb_size:  词向量维度
    :return: 词表、向量表
    """
    vocab = {}  # 词表
    embd = []  # 向量表
    vocab['<unk>'] = 0  # 装载不认识的词
    embd.append([0.001] * emb_size)  # 不认识的词统一为0
    with open(filename, 'r', encoding='UTF-8') as file:
        for idx, line in enumerate(file):
            row = line.strip().split(' ')  # 遍历每行数据，以空格进行拆分
            vocab[row[0]] = idx+1  # 将每个词加入到词表中
            # embd.append(row[1:])
            embedding = [float(val) for val in row[1: ]]
            embd.append(embedding)
    print('Loaded Glove!')
    print("词汇表长度为" + str(len(vocab.keys())))
    return vocab, embd

def testingBatchIter(questions, entitys,label_list, ques_lens, ent_lens):
    """
        返回验证集数据/测试集,逐个获取每一批验证数据的迭代器

        :param question_list: 问题列表 [?,max_len]
        :param entity_list: 答案列表 [?,num_class,maxlen]
        :param label_list: 正确答案所在位置 [?]
        :param ques_lens: 问题长度 [?]
        :param ent_lens: 实体长度 [?,num_class]
    """
    lines = len(questions)
    dataLen = config.batch_size
    batchNum = int(lines / dataLen) + 1
    questions, entitys = np.array(questions), np.array(entitys)
    for batch in range(batchNum):
        startIndex = batch * dataLen
        endIndex = min(batch * dataLen + dataLen, lines)
        returnquestion = []
        for a in questions[startIndex:endIndex]:
            for i in range(config.numclass):
                returnquestion.append(a)
        returnanswer = np.reshape(entitys[startIndex:endIndex], [-1,config.num_step])
        returnquestionlen = []
        for a in ques_lens[startIndex:endIndex]:
            for i in range(config.numclass):
                returnquestionlen.append(a)
        returnanswerlen = np.reshape(ent_lens[startIndex:endIndex],[-1])

        yield np.array(returnquestion), \
              np.array(returnanswer),\
              np.array(label_list[startIndex:endIndex]), \
              np.array(returnquestionlen),\
              np.array(returnanswerlen)

def trainingBatchIter(fb2m_index, trainset):
    """
        逐个获取每一批训练数据的迭代器，会区分每个问题的正确和错误答案，拼接为（q，a+，a-）形式
        :param fb2m_index: 候选三元组列表
        :param questions: 问题列表
        :param answers: 答案列表
        :param labels: 标签列表
    """
    question_list = []
    entity2_list = []
    for a,b,c,d in trainset:
        question_list.append(a)
        entity2_list.append(b+c+d)  # 将三元组合成一个
    batch_size = config.batch_size
    dataLen  = len(question_list)
    batchNum = int(dataLen / batch_size) + 1

    for batch_idx in range(batchNum):
        # 对于每一批问题
        if (batch_idx + 1) * batch_size < len(question_list):
            resultQuestions = question_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            trueAnswers = entity2_list[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        else:  # last batch
            resultQuestions = question_list[batch_idx * batch_size:]
            trueAnswers = entity2_list[batch_idx * batch_size:]
        falseAnswers = [] # 二维矩阵 batchsize * entlen
        for answer in trueAnswers:
            neg = random_neg(fb2m_index, 1, answer)  # 随机产生1个反例
            falseAnswers.append(neg[0][0]+neg[0][1]+neg[0][2])

        ques_lens_list = [len(s) for s in resultQuestions]  # 记录下数据集长度
        false_len_list = []
        for s in falseAnswers:
            if len(s) < config.num_step:
                false_len_list.append(len(s))
            else:
                false_len_list.append(config.num_step)
        true_len_list = []
        for s in trueAnswers:
            if len(s) < config.num_step:
                true_len_list.append(len(s))
            else:
                true_len_list.append(config.num_step)

        # 对数据进行padding操作，使得等长
        ques_pad = [pad_sentences(item, config.num_step) for item in resultQuestions]
        resultQuestions.clear()
        false_pad = [pad_sentences(item, config.num_step) for item in falseAnswers]
        falseAnswers.clear()
        true_pad = [pad_sentences(item, config.num_step) for item in trueAnswers]
        trueAnswers.clear()
        yield np.array(ques_pad), np.array(true_pad), np.array(false_pad), \
              np.array(ques_lens_list), np.array(true_len_list),np.array(false_len_list)

def random_neg(fb2m_name_index, size, anwer):
    """
    # 在fb2m_name_index中随机产生不同于anwer的size个反例
    :param fb2m_name_index:
    :param size:
    :param anwer:
    :return:
    """
    return_answer = []
    while True:
        if len(return_answer) >= size:
            break
        index = random.randint(1, len(fb2m_name_index) - 1)  # fb2m_name_index 第一个元素时{}空元素
        if fb2m_name_index[index] == anwer:
            continue
        if fb2m_name_index[index] not in return_answer:
            return_answer.append(fb2m_name_index[index])
    return return_answer

def pad_sentences(sentence, length, padding_index=0):
    """
    将sentence(以词id)表示，padding到length长度
    :param sentence:
    :param length:
    :param padding_index:
    :return:
    """
    num_padding = length - len(sentence)
    if num_padding >0:
        new_sentence = sentence + [padding_index] * num_padding
    else:
        new_sentence = sentence[:length]  # 太长则进行截取
    return new_sentence