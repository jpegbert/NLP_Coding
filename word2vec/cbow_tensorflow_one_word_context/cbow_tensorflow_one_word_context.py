import numpy as np
from sklearn.preprocessing import  OneHotEncoder
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
采用tensorflow实现cbow的one word context版本
"""

"""
1. 准备训练数据
"""
# Context|Target
corpus_king_queen_symbol = ['king|a', 'queen|a', 'king|b', 'queen|b', 'king|c', 'queen|c', 'king|x',
                            'queen|y', 'man|d', 'woman|d', 'man|e', 'woman|e', 'man|f', 'woman|f',
                            'man|x', 'woman|y']

train_data = [sample.split('|')[0] for sample in corpus_king_queen_symbol]
train_label = [sample.split('|')[1] for sample in corpus_king_queen_symbol]

vocabulary = (list(set(train_data) | set(train_label)))
vocabulary.sort()

one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(np.reshape(vocabulary, (-1, 1)))

X = one_hot_encoder.transform(np.reshape(train_data, (-1, 1))).toarray()
y = one_hot_encoder.transform(np.reshape(train_label, (-1, 1))).toarray()

"""
2. 构建模型
输入是X，y
"""
N = 5
V = len(vocabulary)

inputs = Input(shape=(V, ))
x = Dense(N, activation='linear', use_bias=False)(inputs)
predictions = Dense(V, activation='softmax', use_bias=False)(x)

model = Model(inputs=inputs, outputs=predictions)
model.summary()

"""
3. 训练模型
"""
model.compile(optimizer=keras.optimizers.Adagrad(0.07), loss='categorical_crossentropy', metrics=['accuracy', 'mse'])
model.fit(X, y, batch_size=1, epochs=1000)

"""
4. 验证/可视化结果
"""
weights = model.get_weights()
embeddings = np.array(weights[0])
assert (embeddings.shape == (V, N))
word_vec = dict((word, vector) for word, vector in zip(vocabulary, embeddings))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)
print(X_pca)

fig, ax = plt.subplots()
for i in range(len(X_pca)):
    team = X_pca[i]
    ax.scatter(team[0], team[1])
    ax.annotate(vocabulary[i], (team[0], team[1]))
plt.show()
