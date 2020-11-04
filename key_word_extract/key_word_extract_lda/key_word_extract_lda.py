import math
import numpy as np
from pyhanlp import *
import functools
from gensim import corpora, models


def get_stopword_list():
    """
    加载停用词表
    """
    stop_word_path = 'stopwords.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path).readlines()]
    return stopword_list


def seg_to_list(sentence, pos=False):
    """
    定义分词方法
    """
    seg_list = HanLP.newSegment("perceptron").seg(sentence)
    return seg_list


def word_filter(seg_list, pos=False):
    """
    定义干扰词过滤方法：根据分词结果对干扰词进行过滤
    """
    stopword_list = get_stopword_list()
    filter_list = [str(s.word) for s in seg_list if not s.word in stopword_list and len(s.word) > 1]
    return filter_list


def load_data(pos=False):
    """
    加载数据集，对数据集中的数据分词和过滤干扰词，每个文本最后变成一个非干扰词组成的词语列表
    """
    doc_list = []
    ll = []
    for line in open('corpus.txt', 'r', encoding='utf-8'):
        ll.append(line.strip())
    content = " ".join(ll)
    seg_list = seg_to_list(content, pos)
    filter_list = word_filter(seg_list, pos)
    doc_list.append(filter_list)
    return doc_list


if __name__ == '__main__':
    text = '会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。'
    pos = False
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)
    print('LDA模型结果:')
    topic_extract(filter_list, 'LDA', pos)

