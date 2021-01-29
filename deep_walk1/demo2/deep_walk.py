import numpy as np
import gensim


def deepwalk(_g, _corpus_num):
    _corpus = []
    for i in range(_corpus_num):
        sentence = ['o']  # 'o'为源节点 's'表示汇节点
        current_word = 'o'
        while current_word != 's':
            _node_list = []
            _weight_list = []
            for _nbr, _data in _g[current_word].items():
                _node_list.append(_nbr)
                _weight_list.append(_data['weight'])
            _ps = [float(_weight) / sum(_weight_list) for _weight in _weight_list]
            sel_node = roulette(_node_list, _ps)
            sentence.append(sel_node)
            current_word = sel_node
        _corpus.append(sentence)
    return _corpus


def roulette(_datas, _ps):
    """
    轮盘赌模型-按概率选择指定区域
    :param _datas:
    :param _ps:
    :return:
    """
    return np.random.choice(_datas, p=_ps)


def word_sequence_2_vec(_word_sequence, _dim):
    """
    word2vec训练词向量
    :param _word_sequence:
    :param _dim:
    :return:
    """
    model = gensim.models.Word2Vec(_word_sequence, min_count=1, size=_dim)
    global NODES
    NODES = model.wv.vocab.keys()
    _em_vec = []
    for _node in NODES:
        _em_vec.append(model[_node].tolist())
    global NODES_NUM
    NODES_NUM = len(NODES)
    return np.mat(_em_vec)

