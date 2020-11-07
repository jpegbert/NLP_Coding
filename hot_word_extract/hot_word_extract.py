import glob
from random import choice
import jieba


"""
采用高频词的方式得到热词
"""


def get_content(path):
    """ 数据读取 """
    content = ''
    with open(path, 'r', encoding='gbk', errors='ignore') as rf:
        for line in rf:
            content += line.strip()
    return content


def get_tf(words, topK=10):
    """ 定义高频词统计函数 """
    tf_dic = dict()
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    return sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[:topK]


def demo_topk():
    files = glob.glob('.\\data\\news\\*.txt')
    corpus = [get_content(file) for file in files]
    sample = choice(corpus)
    split_words = list(jieba.cut(sample))

    print('样本之一：' + sample + "\n\n")
    print('样本分词效果：' + '/ '.join(split_words) + "\n\n")
    print('样本top(10)：' + str(get_tf(split_words, 10)) + "\n\n")


def get_stopwords(path):
    """ 获取停用词 """
    with open(path, encoding='utf-8') as rf:
        return [line.strip() for line in rf]


def demo_topk_with_stopwords():
    files = glob.glob('.\\data\\news\\*.txt')
    corpus = [get_content(file) for file in files]
    sample = choice(corpus)
    split_words = list(jieba.cut(sample))

    stopwords = get_stopwords('.\\data\\stop_words.utf8')
    split_words_stopwords = [word for word in split_words if word not in stopwords]

    print('样本之一：' + sample + "\n\n")
    print('样本分词效果：' + '/ '.join(split_words) + "\n\n")
    print('样本top(10)：' + str(get_tf(split_words_stopwords, 10)) + "\n\n")


def demo_topk_with_userdict():
    jieba.load_userdict('.\\data\\user_dict.utf8')
    files = glob.glob('.\\data\\news\\*.txt')
    corpus = [get_content(file) for file in files]
    sample = choice(corpus)
    split_words = list(jieba.cut(sample))

    stopwords = get_stopwords('.\\data\\stop_words.utf8')
    split_words_stopwords = [word for word in split_words if word not in stopwords]

    print('样本之一：' + sample + "\n\n")
    print('样本分词效果：' + '/ '.join(split_words) + "\n\n")
    print('样本top(10)：' + str(get_tf(split_words_stopwords, 10)) + "\n\n")


if __name__ == '__main__':
    # demo_topk() # 不添加停用词，得到top k
    # demo_topk_with_stopwords() # 添加停用词，得到top k
    demo_topk_with_userdict()
