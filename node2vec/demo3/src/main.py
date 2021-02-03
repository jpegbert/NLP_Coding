'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016

https://github.com/aditya-grover/node2vec
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import pandas as pd


def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")
	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist', help='Input graph path')
	parser.add_argument('--output', nargs='?', default='emb/karate.emb', help='Embeddings path')
	parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
	parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
	parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')
	parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
	parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
	parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
	parser.add_argument('--p', type=float, default=1,  help='Return hyperparameter. Default is 1.')
	parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
	parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
	parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)
	return parser.parse_args()


def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G


def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.save_word2vec_format(args.output)
	return model


def cos_similarity(v1, v2):
	"""
	计算两个点的cos相似度
	:param v1:
	:param v2:
	:return:
	"""
	return 1 - spatial.distance.cosine(v1, v2)


def cluster_embedding_result(model):
	embedding_node = []
	for i in range(1, 35):
		j = str(i)
		embedding_node.append(model[j])
	# 一共有34个点
	embedding_node = np.matrix(embedding_node).reshape(34, -1)
	y_pred = cluster.k_means(n_clusters=3, random_state=9).fit_predict(embedding_node)
	print(y_pred)


def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph()
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	model = learn_embeddings(walks)

	# 结果展示与可视化
	print(model["17"])
	nx.draw(nx_G, with_labels=True)
	plt.show()

	# 相似节点组
	print(model.similarity("17", "6"))
	print(model.similarity("7", "6"))
	print(model.similarity("7", "5"))

	# 不相似节点组
	print(model.similarity("17", "25"))

	# 找到和节点最相似的一组点
	print(model.wv.most_similar("34"))

	# 相似节点组
	print(cos_similarity(model["17"], model["6"]))
	print(cos_similarity(model["7"], model["6"]))
	print(cos_similarity(model["7"], model["5"]))

	# 不相似节点组
	print(cos_similarity(model["17"], model["25"]))

	# 对embedding之后的结果进行kmeans cluster


if __name__ == "__main__":
	args = parse_args()
	main(args)
