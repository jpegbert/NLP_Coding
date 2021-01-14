from gensim import corpora
from gensim import models


# https://www.jianshu.com/p/f3b92124cd2b
# https://mp.weixin.qq.com/s/CMSkbJlhGMbF5gXP3NndHQ

corpus = [
    'this is the first document',
    'this is the second second document',
    'and the third one',
    'is this the first document'
]
# 分词处理
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

# 得到每个词的id值及词频
# 赋给语料库中每个词(不重复的词)一个整数id
dictionary = corpora.Dictionary(word_list)
new_corpus = [dictionary.doc2bow(text) for text in word_list]
print(new_corpus)
"""
元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
 [(0, 1), (2, 1), (3, 1), (4, 1), (5, 2)],
 [(3, 1), (6, 1), (7, 1), (8, 1)],
 [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]]
"""
# 得到语料库中每个词对应的id
print(dictionary.token2id)
"""
{'document': 0, 'first': 1, 'is': 2, 'the': 3, 'this': 4, 'second': 5, 'and': 6, 'one': 7, 'third': 8}
"""

# 训练gensim模型并且保存它以便后面的使用
# 训练模型并保存
tfidf = models.TfidfModel(new_corpus)
tfidf.save("my_model.tfidf")

# 载入模型
tfidf = models.TfidfModel.load("my_model.tfidf")

# 使用训练好的模型得到单词的tfidf值
tfidf_vec = []
for i in range(len(corpus)):
    string = corpus[i]
    string_bow = dictionary.doc2bow(string.lower().split())
    string_tfidf = tfidf[string_bow]
    tfidf_vec.append(string_tfidf)
print(tfidf_vec)
"""
[[(0, 0.33699829595119235),
  (1, 0.8119707171924228),
  (2, 0.33699829595119235),
  (4, 0.33699829595119235)],
 [(0, 0.10212329019650272),
  (2, 0.10212329019650272),
  (4, 0.10212329019650272),
  (5, 0.9842319344536239)],
 [(6, 0.5773502691896258), 
  (7, 0.5773502691896258), 
  (8, 0.5773502691896258)],
 [(0, 0.33699829595119235),
  (1, 0.8119707171924228),
  (2, 0.33699829595119235),
  (4, 0.33699829595119235)]]
"""
"""
[[(0, 0.33699829595119235), 
(1, 0.8119707171924228), 
(2, 0.33699829595119235), 
(4, 0.33699829595119235)], 
[(0, 0.10212329019650272), 
(2, 0.10212329019650272), 
(4, 0.10212329019650272), 
(5, 0.9842319344536239)], 
[(6, 0.5773502691896258), (7, 0.5773502691896258), (8, 0.5773502691896258)], [(0, 0.33699829595119235), (1, 0.8119707171924228), (2, 0.33699829595119235), (4, 0.33699829595119235)]]
"""


"""
通过上面的计算我们发现这向量的维数和我们语料单词的个数不一致呀，我们要得到的是每个词的tfidf值，为了一探究竟我们再做个小测试
"""
# 随便拿几个单词来测试
string = 'the i first second name'
string_bow = dictionary.doc2bow(string.lower().split())
string_tfidf = tfidf[string_bow]
print(string_tfidf)
"""
[(1, 0.4472135954999579), (5, 0.8944271909999159)]
"""
"""
结论
gensim训练出来的tf-idf值左边是词的id，右边是词的tfidf值
gensim有自动去除停用词的功能，比如the
gensim会自动去除单个字母，比如i
gensim会去除没有被训练到的词，比如name
所以通过gensim并不能计算每个单词的tfidf值
"""
