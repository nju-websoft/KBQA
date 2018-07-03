import tensorflow as tf
from config import config
from tensorflow import nn
from tensorflow.contrib import rnn

class lstmRNN(object):

    def __init__(self, embeddings):
        self.batchSize = config.batch_size  # 批次
        self.num_step = config.num_step  # 步长
        self.embeddings = embeddings  # 词向量
        self.embeddingSize = config.embed_dim  # 词向量维度
        self.hidden_neural_size = config.hidden_neural_size  # 隐藏层单元
        self.margin = config.margin

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        self.inputQuestions = tf.placeholder(tf.int32, shape=[None, self.num_step])  # 问题输入
        self.inputQuestions_len = tf.placeholder(tf.int32,shape=[None])
        self.inputTrueAnswers = tf.placeholder(tf.int32, shape=[None, self.num_step])
        self.inputTrueAnswers_len = tf.placeholder(tf.int32, shape=[None])
        self.inputFalseAnswers = tf.placeholder(tf.int32, shape=[None, self.num_step])
        self.inputFalseAnswers_len = tf.placeholder(tf.int32, shape=[None])

        self.inputTestQuestions = tf.placeholder(tf.int32, shape=[None, self.num_step])  # 测试时输入的batchsize = batchsize*numcalss
        self.inputTestQuestions_len = tf.placeholder(tf.int32, shape=[None])
        self.inputTestAnswers = tf.placeholder(tf.int32, shape=[None, self.num_step])
        self.inputTestAnswers_len = tf.placeholder(tf.int32, shape=[None])
        self.input_y = tf.placeholder(tf.int32,shape=[None])  #batchsize 指示正确答案index

        # 设置word embedding层
        with tf.name_scope("embedding_layer"):
            tfEmbedding = tf.Variable(self.embeddings, trainable=True, name="W",dtype=tf.float32)  # 词向量
            questions = tf.nn.embedding_lookup(tfEmbedding, self.inputQuestions)  # 问题
            trueAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTrueAnswers)  # 正确的三元组
            falseAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputFalseAnswers)

            testQuestions = tf.nn.embedding_lookup(tfEmbedding, self.inputTestQuestions)
            testAnswers = tf.nn.embedding_lookup(tfEmbedding, self.inputTestAnswers)

        # 建立LSTM网络
        with tf.variable_scope("LSTM_scope", reuse=None):
            self.question1 = self.biLSTMCell(questions, self.inputQuestions_len, self.hidden_neural_size)  # shape = (bitchsize,num_step,n_hidden)
            question2 = tf.nn.tanh(self.max_pooling(self.question1))
        with tf.variable_scope("LSTM_scope", reuse=True):  # 重用LSTM权重
            self.trueAnswer1 = self.biLSTMCell(trueAnswers, self.inputTrueAnswers_len, self.hidden_neural_size)
            trueAnswer2 = tf.nn.tanh(self.max_pooling(self.trueAnswer1))
            self.falseAnswer1 = self.biLSTMCell(falseAnswers, self.inputFalseAnswers_len, self.hidden_neural_size)
            falseAnswer2 = tf.nn.tanh(self.max_pooling(self.falseAnswer1))

            testQuestion1 = self.biLSTMCell(testQuestions, self.inputTestQuestions_len, self.hidden_neural_size)
            testQuestion2 = tf.nn.tanh(self.max_pooling(testQuestion1))
            testAnswer1 = self.biLSTMCell(testAnswers, self.inputTestAnswers_len, self.hidden_neural_size)
            testAnswer2 = tf.nn.tanh(self.max_pooling(testAnswer1))

        self.trueCosSim = self.getCosineSimilarity(question2, trueAnswer2)  # 计算问题和正确三元组之间的余弦值
        self.falseCosSim = self.getCosineSimilarity(question2, falseAnswer2)
        self.loss = self.getLoss(self.trueCosSim, self.falseCosSim, self.margin)  # 计算loss值

        self.result = self.getCosineSimilarity(testQuestion2, testAnswer2)  # shape = [batchsize*numcalss]

        # Calculate Accuracy  # 测试时使用
        with tf.name_scope("accuracy"):
            result = tf.reshape(self.result,[-1,config.numclass])
            self.predictions = tf.argmax(result, 1, output_type=tf.int32)  # shape->(bitchsize,)
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, "float"), name="correct_num")

            # 计算MRR
            ranks = tf.nn.top_k(result, k=config.numclass)  # 对每一个问题的候选答案进行排序
            # print("DEBUG: ranks -> %s" % ranks[1])
            input = tf.reshape(self.input_y,[-1,1])
            input_y = None
            for i in range(config.numclass):
                if i == 0:
                    input_y = input
                else:
                    input_y = tf.concat([input_y, input], 1)
            true_rank = tf.slice(tf.where(tf.equal(ranks[1], input)), [0, 1], [-1, 1])
            self.rr = tf.reduce_sum(tf.divide(1, tf.add(true_rank, 1)))
            # print("DEBUG: ranks -> %s" % self.rr)

    @staticmethod
    def biLSTMCell(x, x_lens, hiddenSize):
        # 前向lstm单元
        lstm_fw_cell = rnn.LSTMCell(hiddenSize)
        # 后向lstm单元
        lstm_bw_cell = rnn.LSTMCell(hiddenSize)
        # 双向lstm
        outputs, output_states = nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length=x_lens, dtype=tf.float32)
        output_fw, output_bw = outputs
        return tf.concat([output_fw,output_bw],1)  # outputs ->(output_fw, output_bw)->(batch_size, max_time, n_hidden)

    @staticmethod
    def getCosineSimilarity(q, a):
        epsilon = 1e-6
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul = tf.reduce_sum(tf.multiply(q, a), 1)
        cosSim = tf.maximum(tf.div(mul, tf.maximum(tf.multiply(q1, a1),epsilon)),epsilon)
        return cosSim

    @staticmethod
    def max_pooling(lstm_out):
        height = int(lstm_out.get_shape()[1])  # num_step
        width = int(lstm_out.get_shape()[2])  # n_hidden
        lstm_out = tf.expand_dims(lstm_out, -1)# shape = (bitchsize,num_step,n_hidden,1)
        output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, width])
        return output

    @staticmethod
    def getLoss(trueCosSim, falseCosSim, margin):
        zero = tf.fill(tf.shape(trueCosSim), 0.0)
        tfMargin = tf.fill(tf.shape(trueCosSim), margin)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(tfMargin, tf.subtract(trueCosSim, falseCosSim)))
            loss = tf.reduce_sum(losses)
        return loss
