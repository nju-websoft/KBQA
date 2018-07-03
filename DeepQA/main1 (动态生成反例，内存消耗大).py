import os
import pickle
import tensorflow as tf
import data_loader
from config import config
from lstmRNN import lstmRNN
import numpy as np
from tqdm import tqdm  # 进度条工具，展示训练进程

def restore():
    try:
        print("正在加载模型，大约需要一分钟...")
        saver.restore(sess, config.trainedModel)
    except Exception as e:
        print(e)
        print("加载模型失败，重新开始训练")
        train()

def train():
    print("重新训练，请保证计算机拥有至少7G空闲内存")
    # 准备训练数据
    print("正在加载训练数据...")
    with open("./data/freebase/fb2m_index.pkl", "rb") as f:  # 加名字index字典
         fb2m_index = list(pickle.load(f))
    with open("./data/SimpleQuestion/traindata.pkl", "rb") as f:  # 加载训练数据
        trainset = list(pickle.load(f))  # 训练集大小75618
    print("加载完成！")
    print("正在加载验证数据...")
    with open(config.developFile, "rb") as f:  # 加载验证数据
        dev_sample = list(pickle.load(f))  # [ques_pad, ans_pad, answer_index, question_len, can_entity_len]
    devsize = len(dev_sample[0])  # 验证集大小
    # 开始训练
    print("开始训练...")
    sess.run(tf.global_variables_initializer())  # 初始化所有变量
    lr = config.lr
    best_correct = 0
    for i in range(config.lrDownCount):  # 调正学习速率
        optimizer = tf.train.GradientDescentOptimizer(lr)  # 梯度下降算法的优化器
        optimizer.apply_gradients(zip(grads, tvars))
        trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
        for epoch in range(config.num_epochs):

            tqs, tta, tfa = [], [], []  # 问题、正确三元组、错误三元组
            tqsl, ttal, tfal = [], [], []  # 问题长度、正确三元组长度、错误三元组长度
            for question, trueAnswer, falseAnswer, _a, _b, _c in data_loader.trainingBatchIter(fb2m_index, trainset):   # 获得训练集数据
                tqs.append(question), tta.append(trueAnswer), tfa.append(falseAnswer), tqsl.append(_a), ttal.append(_b), tfal.append(_c)
            for question, trueAnswer, falseAnswer, _a, _b, _c, in tqdm(zip(tqs, tta, tfa, tqsl, ttal, tfal),
                                                    desc="Training epoch "+ str(epoch)+" lr="+str(lr)+" ", total=len(tqs)):
                feed_dict = {
                    lstm.inputQuestions: question,
                    lstm.inputTrueAnswers: trueAnswer,
                    lstm.inputFalseAnswers: falseAnswer,
                    lstm.inputQuestions_len: _a,
                    lstm.inputTrueAnswers_len: _b,
                    lstm.inputFalseAnswers_len: _c,
                    lstm.keep_prob: config.keep_prob
                }
                _, step, loss, = sess.run([trainOp, globalStep, lstm.loss], feed_dict)  # 执行模型训练，获取loss值
                # print("step:", step, "loss:", loss)

            if (epoch + 1) % 20 !=0:  # 每20epoch跑一边验证集，且记录最高的状态
                continue
            _correct_num, _rr = 0, 0
            for question, answer, lable, question_len, answer_len in tqdm( data_loader.testingBatchIter(dev_sample[0], dev_sample[1], dev_sample[2], dev_sample[3],
                                                                dev_sample[4]),desc="验证集测试中》》》", total=devsize / config.batch_size):
                feed_dict = {
                    lstm.inputTestQuestions: question,
                    lstm.inputTestQuestions_len: question_len,
                    lstm.inputTestAnswers: answer,
                    lstm.inputTestAnswers_len: answer_len,
                    lstm.keep_prob: config.keep_prob,
                    lstm.input_y: lable
                }
                _, scores, correct_num, rr = sess.run([globalStep, lstm.result, lstm.correct_num, lstm.rr], feed_dict)
                _correct_num += correct_num
                _rr += rr
            print("dev acc {:g}, dev mrr {:g}".format(_correct_num / devsize, _rr / devsize))
            if _correct_num > best_correct:  # 记录验证集acc最高的模型参数
                best_correct = _correct_num
                saver.save(sess, config.saveFile)
        lr *= config.lrDownRate  # 更新学习速率

if __name__ == '__main__':
    # 读取测试数据
    print("正在载入测试数据...")
    word2idx, embedding = data_loader.load_glove(config.word2vec_file, config.embed_dim)  # 加载词向量文件
    embedding = np.asarray(embedding)  # 转化为ndarray格式
    word2idx.clear()
    with open(config.testingFile, "rb") as f:  # 加载测试数据
        test_sample = list(pickle.load(f))  # [ques_pad, ans_pad, answer_index, question_len, can_entity_len]
    testsize = len(test_sample[0])
    print("测试数据加载完成")
    with tf.Graph().as_default():  # 加载默认图
        with tf.Session().as_default() as sess:
            # 加载LSTM网络
            print("正在加载LSTM网络...")
            globalStep = tf.Variable(0, name="globle_step", trainable=False)
            lstm = lstmRNN(embedding)  # 加载lstm模型
            tvars = tf.trainable_variables()  # 获取所有需要训练的变量
            grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), config.max_grad_norm)  # 梯度截取，防止梯度爆炸
            saver = tf.train.Saver()
            print("加载完成！")

            # 加载模型或训练模型
            if os.path.exists(config.trainedModel + '.index'):
                while True:
                    choice = input("找到已经训练好的模型，是否载入（y/n）")
                    if choice.strip().lower() == 'y':
                        restore()
                        break
                    elif choice.strip().lower() == 'n':
                        train()
                        break
                    else:
                        print("无效的输入！\n")
            else:
                train()

            # 进行测试，输出结果
            print("正在进行测试，大约需要二十分钟...")
            _correct_num, _rr = 0,0
            for question, answer, lable, question_len, answer_len in tqdm(data_loader.testingBatchIter(test_sample[0], test_sample[1],
                            test_sample[2], test_sample[3], test_sample[4]), desc="测试中》》》",total=testsize/config.batch_size):
                feed_dict = {
                        lstm.inputTestQuestions: question,  # 输入每批次的问题
                        lstm.inputTestQuestions_len: question_len,  # 输入问题长度
                        lstm.inputTestAnswers: answer,  # 输入三元组候选集
                        lstm.inputTestAnswers_len:answer_len,
                        lstm.keep_prob: config.keep_prob,
                        lstm.input_y: lable  # 输入正确答案
                }
                _, scores, correct_num, rr = sess.run([globalStep, lstm.result, lstm.correct_num, lstm.rr], feed_dict)  # 执行计算，并获取分数、正确的个数、rr
                _correct_num +=correct_num
                _rr += rr
            print("test acc {:g}, test mrr {:g}".format(_correct_num / testsize, _rr / testsize))  # 打印精确率和MRR
    print("所有步骤完成！程序结束")
