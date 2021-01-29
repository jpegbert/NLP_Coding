import networkx as nx
from deep_walk1.demo2.deep_walk import deepwalk, word_sequence_2_vec

g = nx.DiGraph()
g.add_weighted_edges_from([('o', '1', 80), ('1', '2', 50), ('1', '3', 30), ('2', '4', 20), ('2', '5', 30),
                           ('2', 's', 10), ('3', '2', 10), ('3', 's', 25), ('4', '5', 10), ('4', 's', 10),
                           ('5', '3', 5), ('5', 's', 35)])

dim = 3
num = 10
corpus = deepwalk(g, num)  # num个句子
embed_vec = word_sequence_2_vec(corpus, dim)
print(embed_vec)
"""
[[ 0.10636719  0.11210059  0.13747229]
 [-0.13376723 -0.12343021  0.08402163]
 [ 0.09138694  0.0213544  -0.1063268 ]
 [ 0.00677756  0.15760943 -0.14945041]
 [ 0.16089217 -0.16032197  0.00263322]
 [ 0.06560824 -0.1438026   0.06395027]
 [ 0.16045608 -0.10573091 -0.09284649]]
"""
