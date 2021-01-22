import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
import os


"""
使用gensim训练Item2vec
"""
"""
在gensim中，word2vec 相关的API都在包gensim.models.word2vec中。和算法有关的参数都在类gensim.models.word2vec.Word2Vec中。算法需要注意的参数有：
sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。
size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。
window：即词向量上下文最大距离，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。
sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。
cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。默认值也是1,不推荐修改默认值。
min_count:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为ηη，默认是0.025。
min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
"""


class CBOW:
    def __init__(self, input_file):
        self.model = self.get_train_data(input_file)

    def get_train_data(self, input_file, L=100):
        if not os.path.exists(input_file):
            return
        score_thr = 4.0
        ratingsDF = pd.read_csv(input_file, index_col=None, sep='::', header=None,
                                names=['user_id', 'movie_id', 'rating', 'timestamp'])
        ratingsDF = ratingsDF[ratingsDF['rating'] > score_thr]
        ratingsDF['movie_id'] = ratingsDF['movie_id'].apply(str)
        movie_list = ratingsDF.groupby('user_id')['movie_id'].apply(list).values
        print('training...')
        model = Word2Vec(movie_list, size=L, window=5, sg=0, hs=0, min_count=1, workers=multiprocessing.cpu_count(), iter=10)
        return model

    def recommend(self, movieID, K):
        """
         Args:
             movieID:the movieID to find similar
             K:recom item num
         Returns:
             a dic,key:itemid ,value:sim score
         """
        movieID = str(movieID)
        rank = self.model.most_similar(movieID, topn=K)
        return rank


def use_model(model):
    """
    模型训练完之后使用模型
    """
    # 找出某一个词向量最相近的词集合
    model.wv.similar_by_word('沙瑞金'.decode('utf-8'), topn=100)
    # 看两个词向量的相近程度
    model.wv.similarity('沙瑞金'.decode('utf-8'), '高育良'.decode('utf-8'))
    # 找出不同类的词
    model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split())


if __name__ == '__main__':
    moviesPath = 'data/ml-1m/movies.dat'
    ratingsPath = 'data/ml-1m/ratings.dat'
    usersPath = 'data/ml-1m/users.dat'

    rank = CBOW(ratingsPath).recommend(movieID=1, K=30)
    print('CBOW result', rank)


