import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import gensim
import random
from gensim.models import Word2Vec
import datetime

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
assert gensim.models.word2vec.FAST_VERSION > -1


df_movies = pd.read_csv('data/movies.csv', error_bad_lines=False, encoding="utf-8")
df_ratings = pd.read_csv('data/ratings.csv')

movieId_to_name = pd.Series(df_movies.title.values, index=df_movies.movieId.values).to_dict()
name_to_movieId = pd.Series(df_movies.movieId.values, index=df_movies.title).to_dict()
# print(df_movies.head())
# print(df_ratings.head())

# for key, val in name_to_movieId.items():
#     print(key, val)

plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
ax.set_title("Distribution of Movie Ratings", fontsize=16)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Movie Rating", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.hist(df_ratings['rating'], color="#3F5D7D")
plt.show()

df_ratings_train, df_ratings_test = train_test_split(df_ratings, stratify=df_ratings['userId'], random_state = 15688, test_size=0.30)
print("Number of training data: " + str(len(df_ratings_train)))
print("Number of test data: " + str(len(df_ratings_test)))


"""
为了让模型学习item embedding，需要从数据中获取“单词”和“句子”等价物。在这里，把每个“电影”看做是一个“词”，
并且从用户那里获得相似评级的电影都在同一个“句子”中。
具体来说，“句子”是通过以下过程生成的：为每个用户生成2个列表，分别存储用户“喜欢”和“不喜欢”的电影。
第一个列表包含所有的电影评级为4分或以上。第二个列表包含其余的电影。这些列表就是训练gensim word2vec模型的输入了。
"""


def rating_splitter(df):
    df['liked'] = np.where(df['rating'] >= 4, 1, 0)
    df['movieId'] = df['movieId'].astype('str')
    gp_user_like = df.groupby(['liked', 'userId'])
    return [gp_user_like.get_group(gp)['movieId'].tolist() for gp in gp_user_like.groups]


pd.options.mode.chained_assignment = None
splitted_movies = rating_splitter(df_ratings_train)

# 打乱数据集，因为item2vec数据中item的顺序没有意义
for movie_list in splitted_movies:
    random.shuffle(movie_list)

start = datetime.datetime.now()
"""
对于原来的word2vec，窗口大小会影响我们搜索“上下文”以定义给定单词含义的范围。按照定义，窗口的大小是固定的。但是，
在item2vec实现中，电影的“含义”应该由同一列表中的所有邻居捕获。换句话说，我们应该考虑用户“喜欢”的所有电影，
以定义这些电影的“含义”。这也适用于用户“不喜欢”的所有电影。然后需要根据每个电影列表的大小更改窗口大小。
为了在不修改gensim模型的底层代码的情况下解决这个问题，首先指定一个非常大的窗口大小，这个窗口大小远远大于训练样本中
任何电影列表的长度。然后，在将训练数据输入模型之前对其进行无序处理，因为在使用“邻近”定义电影的“含义”时，
电影的顺序没有任何意义。
Gensim模型中的窗口参数实际上是随机动态的。我们指定最大窗口大小，而不是实际使用的窗口大小。尽管上面的解决方法并不理想，
但它确实实现了可接受的性能。最好的方法可能是直接修改gensim中的底层代码。
"""
model = Word2Vec(sentences=splitted_movies, # We will supply the pre-processed list of moive lists to this parameter
                 iter=5, # epoch
                 min_count=10, # a movie has to appear more than 10 times to be keeped
                 size=200, # size of the hidden layer
                 workers=2, # specify the number of threads to be used for training
                 sg=1, # Defines the training algorithm. We will use skip-gram so 1 is chosen.
                 hs=0, # Set to 0, as we are applying negative sampling.
                 negative=5, # If > 0, negative sampling will be used. We will use a value of 5.
                 window=9999999) #
print("Time passed: " + str(datetime.datetime.now() - start))
model.save('item2vec_model')

"""
模型训练完之后，模型可以保存在您的存储中以备将来使用。注意，gensim保存了所有关于模型的信息，包括隐藏的权重、
词汇频率和模型的二叉树，因此可以在加载文件后继续训练。然而，这是以运行模型时的内存为代价的，因为它将存储在你
的RAM中。如果你只需要隐藏层的权重，它可以从模型中单独提取。下面的代码演示如何保存、加载模型和提取单词向量(embedding)。
"""
# model = Word2Vec.load('item2vec_model')
# word_vectors = model.wv
# del model






