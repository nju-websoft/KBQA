# -*- coding:utf8 -*-
class config(object):
    hidden_neural_size = 75  # 输入问句的lstm隐藏单元的个数
    embed_dim = 50  # 预训练的词向量维度
    keep_prob = 1.0  # dropout rate
    margin = 0.5  # loss 函数中的超参数
    lr = 0.05  # 学习速率
    lrDownRate = 0.5  # 学习速度下降速度
    lrDownCount = 3  # 学习速度下降次数
    batch_size = 100  # 每批次大小，设置的小一点，减少内存
    num_step = 36  # 问句对应的lstm 步长，即问句最长长度
    num_epochs = 40  # 每次学习速度指数下降之前执行的完整epoch次数

    numclass = 80   # 正例和反例的数目比，或者候选集和正确答案之间的比列 训练时1，测试时80
    embeddings_trainable = True
    max_grad_norm = 5  # 设置梯度的最大范数,用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小

    word2vec_file = './data/glove/glove.6B.50d.txt'

    trainingFile = "./data/SimpleQuestion/traindata.pkl"
    developFile = "./data/valid_sample_all.pkl"
    testingFile = "./data/test_sample.pkl"

    saveFile = "newModel/savedModel"
    trainedModel = "trainedModel-50d/savedModel"
