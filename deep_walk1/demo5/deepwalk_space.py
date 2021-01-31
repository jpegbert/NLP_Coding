import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("space_data.tsv", sep="\t")
print(df.head())
print(df.shape)
# construct an undirected graph
G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())
print(len(G)) # number of nodes


def get_randomwalk(node, path_length):
    """
    function to generate random walk sequences of nodes
    :param node:
    :param path_length:
    :return:
    """
    random_walk = [node]

    for i in range(path_length - 1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node

    return random_walk


# get_randomwalk('space exploration', 10)
all_nodes = list(G.nodes())

random_walks = []

for n in tqdm(all_nodes):
    for i in range(5):
        random_walks.append(get_randomwalk(n, 10))

# count of sequences
print(len(random_walks))
# train word2vec model
model = Word2Vec(window=4,
                 sg=1,
                 hs=0,
                 negative=10, # for negative sampling
                 alpha=0.03,
                 min_alpha=0.0007,
                 seed=14)

model.build_vocab(random_walks, progress_per=2)
model.train(random_walks, total_examples=model.corpus_count, epochs=20, report_delay=1)
print(model)
# find top n similar nodes
res = model.similar_by_word('astronaut training')
print(res)

terms = ['lunar escape systems', 'soviet moonshot', 'soyuz 7k-l1', 'moon landing',
         'space food', 'food systems on space exploration missions', 'meal, ready-to-eat',
         'space law', 'metalaw', 'moon treaty', 'legal aspects of computing',
         'astronaut training', 'reduced-gravity aircraft', 'space adaptation syndrome', 'micro-g environment']


def plot_nodes(word_list):
    X = model[word_list]

    # reduce dimensions to 2
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    plt.figure(figsize=(12, 9))
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()


plot_nodes(terms)

