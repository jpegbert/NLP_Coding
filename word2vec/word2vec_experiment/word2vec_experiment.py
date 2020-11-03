import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from gensim.models.word2vec import Word2Vec
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


"""
Word2vec实操  用每日新闻预测金融市场变化
"""


# 读入数据
data = pd.read_csv('../corpus/Combined_News_DJIA.csv')
# print(data.head())

# 可以先把数据给分成Training/Testing data
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
# columns和.index两个属性返回数据集的列索引和行索引
X_train = train[train.columns[2:]]
X_test = test[test.columns[2:]]
# print(X_train)


# flatten转变数据为一维数组就会得到list of sentences。astype强制类型转换
# 同时我们的X_train和X_test可不能随便flatten，他们需要与y_train和y_test对应
corpus = X_train.values.flatten().astype(str)
# print(corpus)
X_train = X_train.values.astype(str)
X_test = X_test.values.astype(str)

X_train = np.array([' '.join(x) for x in X_train])
X_test = np.array([' '.join(x) for x in X_test])

y_train = train['Label'].values
y_test = test['Label'].values
# print(y_train)

# tokenize分割单词
# import nltk
# 如果报错 Resource punkt not found.
# 运行 nltk.download('punkt')
# nltk.download('punkt')
corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]
print(X_train[:2])


def hasNumbers(inputString): # 数字
    return bool(re.search(r'\d', inputString))


def isSymbol(inputString): # 特殊符号
    return bool(re.match(r'[^\w]', inputString))


wordnet_lemmatizer = WordNetLemmatizer()


def check(word, stop=[]):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    word = word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True


def preprocessing(sen): # 把上面的方法综合起来
    res = []
    for word in sen:
        if check(word):
            # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。。之前数据没处理好，其他case里不会有这个情况
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res

"""
若报错Resource wordnet not found.
import nltk
nltk.download('wordnet')
"""
corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]
# print(corpus)
# print(X_train)


"""
训练NLP模型
有了这些干净的数据集，我们可以做我们的NLP模型了。

先用最简单的Word2Vec
"""

model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
# print(model['ok'])

"""
用NLP模型表达我们的数据
接着，我们可以用这个坐标，来表示之前干干净净的数据。
但是有个问题。我们的vec是基于每个单词的，怎么办呢？
由于文本本身的量很小，我们可以把所有的单词的vector拿过来取个平均值
"""
# 先拿到全部的vocabulary
vocab = model.wv.vocab


def get_vector(word_list): # 得到任意text的vector
    # 建立一个全是0的array
    res = np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res / count    # 得到了一个取得任意word list平均vector值


wordlist_train = X_train
wordlist_test = X_test

X_train = [get_vector(x) for x in X_train]
X_test = [get_vector(x) for x in X_test]
# print(X_train[10])

"""
建立ML模型
因为我们128维的每一个值都是连续关系的。所以，不太适合用RandomForest这类把每个column当做单独的variable来看的方法。
"""
params = [0.1, 0.5, 1, 3, 5, 7, 10, 12, 16, 20, 25, 30, 35, 40]
test_scores = []
for param in params:
    clf = SVR(gamma=param)
    test_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
    test_scores.append(np.mean(test_score))

plt.plot(params, test_scores)
plt.title("Param vs CV AUC Score")
plt.show()


"""
用CNN来提升逼格
用vector表示出一个大matrix，并用CNN做“降维+注意力”
（下面内容做的比较简单。要是想更加复杂准确，直接调整参数，往大了调，就行）
首先，我们确定一个padding_size(为了让我们生成的matrix是一样的size)
"""
def transform_to_matrix(x, padding_size=256, vec_size=128): # vec_size 指本身vector的size
    res = []
    for sen in x:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                # 这里有两种except情况，
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res


X_train = transform_to_matrix(wordlist_train)
X_test = transform_to_matrix(wordlist_test)
# print(X_train[123])
"""
可以看到，现在得到的就是一个大大的Matrix，它的size是 128 * 256
每一个这样的matrix，就是对应了我们每一个数据点
在进行下一步之前，我们把我们的input要reshape一下。
原因是要让每一个matrix外部“包裹”一层维度。告诉CNN model，每个数据点都是独立的。之间没有前后关系。
"""
# 转换成np的数组，便于处理
X_train = np.array(X_train)
X_test = np.array(X_test)
# 看看数组的大小
# print(X_train.shape) # (1611, 256, 128)
# print(X_test.shape) # (378, 256, 128)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
# print(X_train.shape) # (1611, 1, 256, 128)
# print(X_test.shape) # (378, 1, 256, 128)


"""
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras import backend as K
# K.set_image_dim_ordering("th")
K.image_data_format() == 'channels_last'
# K.set_image_data_format('channels_last')

# set parameters:
batch_size = 32
n_filter = 16
filter_length = 4
nb_epoch = 5
n_pool = 2

# 新建一个sequential的模型
model = Sequential()
model.add(Convolution2D(n_filter, filter_length, filter_length, input_shape=(1, 256, 128)))
model.add(Activation('relu'))
model.add(Convolution2D(n_filter, filter_length, filter_length))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
# 后面接上一个ANN
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
# compile模型
model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
"""


