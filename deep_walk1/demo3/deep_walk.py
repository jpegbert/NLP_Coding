import pandas as pd
import networkx as nx
import random
from tqdm import tqdm
from gensim.models import Word2Vec


def get_randomwalk(G, node, path_length):
    """
    function to generate random walk sequences of nodes
    :param node:
    :param path_length:
    :return:
    """
    random_walk = [node]
    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        # 邻接节点随机取一个
        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
    return random_walk


def main():
    df = pd.read_csv("space_data.tsv", sep="\t")
    # 链接depth作为边的权重
    G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())
    all_nodes = list(G.nodes())
    random_walks = []
    for n in tqdm(all_nodes):
        for i in range(5):  # 一共更新5次
            random_walks.append(get_randomwalk(G, n, 10))
    model = Word2Vec(window=4,
                     sg=1,
                     hs=0,
                     negative=10, # for negative sampling
                     alpha=0.03,
                     min_alpha=0.0007,
                     seed=14)
    model.build_vocab(random_walks, progress_per=2)
    model.train(random_walks, total_examples=model.corpus_count, epochs=20, report_delay=1)


if __name__ == '__main__':
    main()
