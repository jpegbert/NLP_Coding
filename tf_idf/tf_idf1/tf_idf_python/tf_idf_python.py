from collections import Counter
import math


corpus = [
    'this is the first document',
    'this is the second second document',
    'and the third one',
    'is this the first document'
]

# 对语料进行分词
word_list = []
for i in range(len(corpus)):
    word_list.append(corpus[i].split(' '))
print(word_list)
"""
[['this', 'is', 'the', 'first', 'document'],
 ['this', 'is', 'the', 'second', 'second', 'document'],
 ['and', 'the', 'third', 'one'],
 ['is', 'this', 'the', 'first', 'document']]
"""

# 统计词频
countlist = []
for i in range(len(word_list)):
    count = Counter(word_list[i])
    countlist.append(count)
print(countlist)
"""
[Counter({'this': 1, 'is': 1, 'the': 1, 'first': 1, 'document': 1}), 
Counter({'second': 2, 'this': 1, 'is': 1, 'the': 1, 'document': 1}), 
Counter({'and': 1, 'the': 1, 'third': 1, 'one': 1}), 
Counter({'is': 1, 'this': 1, 'the': 1, 'first': 1, 'document': 1})]
"""


"""
定义计算tfidf公式的函数
"""
def tf(word, count):
    """
    word可以通过count得到，count可以通过countlist得到
    count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
    """
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    """
    统计含有该单词的句子数
    """
    return sum(1 for count in count_list if word in count)


def idf(word, count_list):
    """
    len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
    """
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))


def tfidf(word, count, count_list):
    """
    将tf和idf相乘
    """
    return tf(word, count) * idf(word, count_list)


# 计算每个单词的tfidf值
for i, count in enumerate(countlist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, count, countlist) for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
"""
Top words in document 1
    Word: first, TF-IDF: 0.05754
    Word: this, TF-IDF: 0.0
    Word: is, TF-IDF: 0.0
    Word: document, TF-IDF: 0.0
    Word: the, TF-IDF: -0.04463
Top words in document 2
    Word: second, TF-IDF: 0.23105
    Word: this, TF-IDF: 0.0
    Word: is, TF-IDF: 0.0
    Word: document, TF-IDF: 0.0
    Word: the, TF-IDF: -0.03719
Top words in document 3
    Word: and, TF-IDF: 0.17329
    Word: third, TF-IDF: 0.17329
    Word: one, TF-IDF: 0.17329
    Word: the, TF-IDF: -0.05579
Top words in document 4
    Word: first, TF-IDF: 0.05754
    Word: is, TF-IDF: 0.0
    Word: this, TF-IDF: 0.0
    Word: document, TF-IDF: 0.0
    Word: the, TF-IDF: -0.04463
"""

"""
一般处理方法是把句子里涉及到的单词用word2vec模型训练得到词向量，然后把这些向量加起来再除以单词数，就可以得到句子向量。
这样处理之后可以拿去给分类算法(比如LogisticRegression)训练，从而对文本进行分类。
还有一种是把句子里的每个单词的向量拼接起来，比如每个单词的维度是1X100
一句话有30个单词，那么如何表示这句话的向量呢？
把单词拼接来，最终得到这句话的向量的维度就是30X100维
我想做的是把句子里所有的单词用word2vec模型训练得到词向量，然后把这些向量乘以我们之前得到的tfidf值，
再把它们加起来除以单词数，就可以得到句子向量。也就是结合tfidf给单词加上一个权重，评判一个单词的重要程度。
"""

