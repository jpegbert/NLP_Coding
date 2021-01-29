import random
import networkx as nx
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


random.seed(666)


def random_walk(G, start=None, path_length=20, alpha=0, rand=random.Random()):
    """
    return a random walk path
    """
    if start:
        path = [start]
    else:
        path = [random.choice(list(G.nodes()))]
    while len(path) < path_length:
        curr = path[-1]
        # find it's neighbors
        if len(G[curr]) > 0:
            if rand.random() >= alpha:
                path.append(rand.choice(list(nx.all_neighbors(G, curr))))
            else:
                path.append(path[0])
        else:
            break
    return path


def build_graph():
    node_list = ["friend1", "friend2", "friend3", "Me", "Zhang", "Lan", "friend4", "friend5"]
    edge_list = [("friend1", "Me", 1), ("friend2", "Me", 1), ("friend3", "Me", 1), ("Zhang", "Me", 1), ("Lan", "Me", 1),
                 ("friend4", "Zhang", 1), ("friend5", "Lan", 1), ("Zhang", "Lan", 1)]
    # 创建空图
    G = nx.Graph()
    # 从一个列表中添加节点
    G.add_nodes_from(node_list)
    # 根据(边，边，权重)加载
    G.add_weighted_edges_from(edge_list)
    # plot
    nx.draw_networkx(G)
    plt.show()
    return G


def build_deep_walk_corpus(G, num_paths):
    """
    建立随机游走语料库
    """
    walks = []
    nodes = list(G.nodes)
    for i in range(num_paths):
        for node in nodes:
            walks.append(random_walk(G, start=node))
    return walks


def main():
    G = build_graph()
    print(random_walk(G, start="Me"))
    print(build_deep_walk_corpus(G, num_paths=2))
    corpus = build_deep_walk_corpus(G, num_paths=20)
    model = Word2Vec(corpus, size=20, window=2, min_count=1, sg=1, iter=30)
    print(model.wv["Me"])
    # 与上面一种方式效果一样
    print(model["Me"])
    print(model.wv.most_similar(["friend4"]))


if __name__ == '__main__':
    main()

