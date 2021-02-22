import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

"""
使用gensim训练word2vec模型
主要是记录gensim库训练和使用词向量的相关方法
"""


vocab = 'vocab.txt'
model = Word2Vec(sentences=LineSentence(vocab), # 输入的序列数据
                 size=32, # 词向量的维度，默认为100
                 alpha=0.025, # 初始学习率，默认是0.025
                 min_alpha=0.0001, # 最小学习率，在模型训练过程中逐步降低到这个最小值
                 min_count=5, # 最小词频，低于min_count的词将被删除，默认是5
                 window=5, # 窗口大小，默认为5
                 max_vocab_size=None, # 词向量模型最大允许不重复词的数量，每1千万个词大概需要1G RAM
                 sample=1e-3, # 下采样阈值，对于高频的词将给随机下采样，通常设置为0 到 1e-5之间的值
                 workers=multiprocessing.cpu_count(), # 使用几个cpu训练模型
                 sg=1, # 选用那种词向量算法，skip-gram为1，CBOW为0，默认是0
                 hs=0, # 选用哪种加速训练算法，1表示hierarchical softmax，设置为0表示negative sampling
                 negative=5, # 是否采用负采样，大于0表示使用负采样，其值表示负采样词的个数，等于0表示不采样负采样，默认是5
                 iter=15 # 模型训练迭代轮数，默认是5
)
# 保存模型
model.save('model/word2vec.model')
# 或者使用下面的方式保存模型
# model.wv.save('model/word2vec.model')
# 加载使用模型
model = Word2Vec.load('model/word2vec.model')

# 查看字典
print(model.wv.vocab)
# 查询某一单词对应的向量
print(model.wv["中国"])
# 查询最相似的单词
print(model.most_similar("军事"))
print(model.wv.most_similar("军事"))
print(model.most_similar_cosmul("军事")) # 余弦相似度
# 关联相似度查询
print(model.wv.most_similar(positive=["军事", "政治"], negative=["经济"]))
# 直接查询相似程度
print(model.wv.similarity("男人", "军事"))
print(model.similarity("男人", "军事"))
# 查找异类词
print(model.wv.doesnt_match(["中国", "美国", "叙利亚", "水果"]))

