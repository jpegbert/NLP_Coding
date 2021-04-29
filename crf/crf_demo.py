import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""
https://www.cnblogs.com/pinking/p/9194865.html
https://www.cnblogs.com/pinking/p/8531405.html
"""


TIME_STEPS = 15  # 20 # backpropagation through time 的time_steps
BATCH_SIZE = 1  # 50
INPUT_SIZE = 1  # x数据输入size
LR = 0.05  # learning rate
num_tags = 2


# 定义一个生成数据的 get_batch function:
def get_batch():
    xs = np.array([[2, 3, 4, 5, 5, 5, 1, 5, 3, 2, 5, 5, 5, 3, 5]])
    res = np.array([[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]])
    return [xs[:, :, np.newaxis], res]


# 定义 CRF 的主体结构
class CRF(object):
    def __init__(self, n_steps, input_size, num_tags, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.num_tags = num_tags
        self.batch_size = batch_size
        self.xs = tf.placeholder(tf.float32, [None, self.n_steps, self.input_size], name='xs')
        self.ys = tf.placeholder(tf.int32, [self.batch_size, self.n_steps], name='ys')
        # 将输入 batch_size x seq_length x input_size   映射到 batch_size x seq_length x num_tags

        weights = tf.get_variable("weights", [self.input_size, self.num_tags])
        matricized_x_t = tf.reshape(self.xs, [-1, self.input_size])
        matricized_unary_scores = tf.matmul(matricized_x_t, weights)
        unary_scores = tf.reshape(matricized_unary_scores, [self.batch_size, self.n_steps, self.num_tags])

        sequence_lengths = np.full(self.batch_size, self.n_steps, dtype=np.int32)

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, self.ys, sequence_lengths)

        self.pred, viterbi_score = tf.contrib.crf.crf_decode(unary_scores, transition_params, sequence_lengths)
        # add a training op to tune the parameters.
        self.cost = tf.reduce_mean(-log_likelihood)
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)


# 训练 CRF
if __name__ == '__main__':

    # 搭建 CRF 模型
    model = CRF(TIME_STEPS, INPUT_SIZE, num_tags, BATCH_SIZE)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # matplotlib可视化
    plt.ion()  # 设置连续 plot
    plt.show()
    # 训练多次
    for i in range(150):
        xs, res = get_batch()  # 提取 batch data
        # print(res.shape)
        # 初始化 data
        feed_dict = {
            model.xs: xs,
            model.ys: res,
        }
        # 训练
        _, cost, pred = sess.run(
            [model.train_op, model.cost, model.pred],
            feed_dict=feed_dict)

        # plotting

        x = xs.reshape(-1, 1)
        r = res.reshape(-1, 1)
        p = pred.reshape(-1, 1)

        x = range(len(x))

        plt.clf()
        plt.plot(x, r, 'r', x, p, 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)  # 每 0.3 s 刷新一次

        # 打印 cost 结果
        if i % 20 == 0:
            print('cost: ', round(cost, 4))
