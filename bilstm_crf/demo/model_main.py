from bilstm_crf.demo.model_get import modelConfig, BiLSTM_CRF
import tensorflow as tf
from bilstm_crf.demo import utils

"""
https://mp.weixin.qq.com/s/VhHJHUPfkHVu3rlmpvbUhQ
https://gitee.com/chashaozgr/noteLibrary/tree/master/nlp_trial/ner/src/bilstm_crf
"""


# PATH
w2id_PATH = "./data/word2id_dict"
id2w_PATH = "./data/id2word_dict"
t2id_PATH = "./data/tag2id_dict"
id2t_PATH = "./data/id2tag_dict"
x_train_PATH = "./data/x_train"
y_train_PATH = "./data/y_train"
x_valid_PATH = "./data/x_valid"
y_valid_PATH = "./data/y_valid"
x_test_PATH = "./data/x_test"
y_test_PATH = "./data/y_test"
SEQ_LEN = 10


# 数据加载
x_train = utils.load_dataset(x_train_PATH, pad_len=SEQ_LEN)
y_train = utils.load_dataset(y_train_PATH, pad_len=SEQ_LEN)
x_valid = utils.load_dataset(x_valid_PATH, pad_len=SEQ_LEN)
y_valid = utils.load_dataset(y_valid_PATH, pad_len=SEQ_LEN)
x_test = utils.load_dataset(x_test_PATH, pad_len=SEQ_LEN)
y_test = utils.load_dataset(y_test_PATH, pad_len=SEQ_LEN)

# 字典加载
t2id_dict = utils.load_2id_dic(t2id_PATH)
w2id_dict = utils.load_2id_dic(w2id_PATH)
id2w_dict = utils.load_id2_dic(id2w_PATH)
id2t_dict = utils.load_id2_dic(id2t_PATH)

# 数据转化
x_train = utils.item2id_batch(x_train, w2id_dict)
y_train = utils.item2id_batch(y_train, t2id_dict)
x_valid = utils.item2id_batch(x_valid, w2id_dict)
y_valid = utils.item2id_batch(y_valid, t2id_dict)
x_test = utils.item2id_batch(x_test, w2id_dict)
y_test = utils.item2id_batch(y_test, t2id_dict)

# 模型初始化
modelConf = modelConfig()
modelConf.seq_length = len(x_train[-1][-1])      # 序列长度
modelConf.num_classes = len(t2id_dict)           # 类别数
modelConf.batch_size = len(x_train[-1])          # 每批训练大小
modelConf.num_batches = len(x_train)             # 一共有多少batch
modelConf.vocab_size = len(w2id_dict)            # 词汇量
modelConf.num_epochs = 20                        # 迭代代数
model = BiLSTM_CRF(modelConf)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(modelConf.num_epochs):
        tmp_batch_id = 0
        # training
        while tmp_batch_id < len(x_train):
            sess.run(model.train, feed_dict={"input_x:0": x_train[tmp_batch_id],
                                             "input_y:0": y_train[tmp_batch_id],
                                             "keep_prob:0": modelConf.keep_prob})
            tmp_batch_id = tmp_batch_id + 1
        loss = sess.run(model.loss, feed_dict={"input_x:0": x_train[0],
                                               "input_y:0": y_train[0],
                                               "keep_prob:0": modelConf.keep_prob})
        # validating
        tmp_batch_id = 0
        y_pred = []
        y_valid_combine = []
        while tmp_batch_id < len(x_valid):
            y_pred_batch = sess.run(model.viterbi_sequence, feed_dict={"input_x:0": x_valid[tmp_batch_id],
                                                                       "input_y:0": y_valid[tmp_batch_id],
                                                                       "keep_prob:0": modelConf.keep_prob})
            for idx in range(len(y_pred_batch)):
                y_pred = y_pred + y_pred_batch[idx].tolist()
                y_valid_combine = y_valid_combine + y_valid[tmp_batch_id][idx]
            tmp_batch_id = tmp_batch_id + 1
        p, r, f1score = utils.model_rep(y_pred, y_valid_combine)
        print("epoch: %s, loss:%s, precision:%s, recall:%s, f1: %s" % (i, loss, p, r, f1score))
        utils.print_matrix(utils.model_conf(y_pred, y_valid_combine))
        print("-----------------------------")
