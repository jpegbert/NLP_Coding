import tensorflow as tf
import numpy as np


class modelConfig(object):
    """模型必要参数"""
    embedding_dim = 300  # 词向量维度
    seq_length = 20  # 序列长度
    num_classes = 11  # 类别数
    # hidden_dim = 64  # 全连接层神经元

    keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-4  # 学习率

    batch_size = 64  # 每批训练大小
    num_batches = 263  # 一共有多少batch
    num_epochs = 20  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果


class BiLSTM_CRF(object):
    """BiLSTM_CRF 命名实体识别"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.bilstm_crf()

    def bilstm_crf(self):
        with tf.name_scope("embedding"):
            # embedding layer
            w2v_matrix = tf.get_variable(name="w2v_matrix", shape=[self.config.vocab_size, self.config.embedding_dim],
                                         dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            embedding_inputs = tf.nn.embedding_lookup(w2v_matrix, self.input_x)
            embedding_inputs = tf.nn.dropout(embedding_inputs, self.keep_prob)

        with tf.name_scope("BiLSTM"):
            # BiLSTM layer
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(100, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(100, forget_bias=1.0, state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                             lstm_bw_cell,
                                                                             embedding_inputs,
                                                                             dtype=tf.float32,
                                                                             time_major=False,
                                                                             scope=None)
            bilstm_out = tf.concat([output_fw, output_bw], axis=2)
            self.bilstm_tmp = bilstm_out

        with tf.name_scope("dense"):
            # dense layer
            # W = tf.Variable(tf.truncated_normal([2 * 100, self.config.num_classes],stddev=0.1))
            # b = tf.Variable(tf.truncated_normal([self.config.num_classes],stddev=0.1))
            # reshape_bilstm_out = tf.reshape(bilstm_out, [-1, 2*100])
            # dense_out = tf.tanh(tf.matmul(reshape_bilstm_out, W) + b)
            # dense_out_reshape = tf.reshape(dense_out, [self.config.batch_size, -1, self.config.num_classes])
            W = tf.get_variable(name="W_dense", shape=[self.config.batch_size, 2 * 100, self.config.num_classes],
                                dtype=tf.float32, initializer=tf.truncated_normal_initializer())
            b = tf.get_variable(name="b_dense", shape=[self.config.batch_size, self.config.seq_length, self.config.num_classes],
                                dtype=tf.float32, initializer=tf.zeros_initializer())
            dense_out = tf.tanh(tf.matmul(bilstm_out, W) + b)

        with tf.name_scope("crf"):
            # CRF
            sequence_lengths = np.full(self.config.batch_size, self.config.seq_length, dtype=np.int32)
            self.shape1 = sequence_lengths
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(dense_out, self.input_y, sequence_lengths)
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(dense_out, self.transition_params, sequence_lengths)

        self.loss = tf.reduce_mean(-log_likelihood)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train = optimizer.minimize(self.loss)



# # 单测
# input_x = [[0, 1, 2], [2, 3, 4]]
# input_y = [[1, 1, 0], [2, 2, 1]]
# model_config = modelConfig()
# model_config.batch_size = 2
# model_config.embedding_dim = 5
# model_config.num_classes = 3
# model_config.seq_length = 3
# model_config.vocab_size = 5

# model = BiLSTM_CRF(model_config)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     # print(shape1)
#     crf_out = sess.run([model.viterbi_sequence, model.viterbi_score], feed_dict={
#                               "input_x:0": input_x, "input_y:0": input_y, "keep_prob:0": model_config.keep_prob})
#     loss_out = sess.run([model.loss], feed_dict={
#         "input_x:0": input_x, "input_y:0": input_y, "keep_prob:0": model_config.keep_prob})
#     print(loss_out)
#     print(crf_out)
#     for i in range(500):
#         sess.run(model.train, feed_dict={
#             "input_x:0": input_x, "input_y:0": input_y, "keep_prob:0": model_config.keep_prob})
#     crf_out = sess.run([model.viterbi_sequence, model.viterbi_score], feed_dict={
#                               "input_x:0": input_x, "input_y:0": input_y, "keep_prob:0": model_config.keep_prob})
#     loss_out = sess.run([model.loss], feed_dict={
#         "input_x:0": input_x, "input_y:0": input_y, "keep_prob:0": model_config.keep_prob})
#     print(loss_out)
#     print(crf_out)

