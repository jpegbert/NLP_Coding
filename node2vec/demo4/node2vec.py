from joblib import Parallel, delayed
from collections import defaultdict, OrderedDict
import numpy as np
from node2vec.demo4.util import read_graph_data, partition_num


class Node2Vec:
    def __init__(self, data_cate_path, data_link_path, walk_per_vertex,
                 walk_length, p, q, n_workers=1, verbose=50, **kargs):
        self.vertex, self.edge = read_graph_data(data_cate_path, data_link_path)
        self.walk_per_vertex = walk_per_vertex  # r defined in paper
        self.walk_length = walk_length  # l defined in paper
        self.inverse_p = 1.0 / p  # p defined in paper
        self.inverse_q = 1.0 / q  # q defined in paper
        self.vertex_list = sum(list(self.vertex.values()), [])

        self.process_modified_weights()
        self.run(walk_per_vertex=walk_per_vertex, n_workers=n_workers, verbose=verbose)

    def process_modified_weights(self): # 这里默认把边权重都看作是1了，如果需要权重，可以修改权重变量
        self.second_markov_weight_dict = defaultdict(dict)  # pi defined in paper
        for pre_one in self.vertex_list:
            for curr_one in self.edge[pre_one]:
                weight_dict = OrderedDict()
                for next_one in self.edge[curr_one]:
                    if next_one not in self.edge[pre_one]:
                        weight_dict[next_one] = self.inverse_q  # distance = 2
                    elif next_one == pre_one:
                        weight_dict[next_one] = self.inverse_p  # distance = 0
                    else:
                        weight_dict[next_one] = 1  # distance = 1

                item_list = list(weight_dict.keys())
                value_list = list(weight_dict.values())
                value_list = list(map(lambda x: x / sum(value_list), value_list))

                self.second_markov_weight_dict[pre_one][curr_one] = (item_list, value_list)

    def node2vec(self, walk_per_vertex, walk_length):
        res = []
        for _ in range(walk_per_vertex):
            for start_node in self.vertex_list:
                walk_path = [start_node]
                if not self.edge[start_node]:
                    continue
                else:
                    walk_path.append(np.random.choice(self.edge[start_node]))

                for _ in range(2, walk_length):
                    pre_one, curr_one = walk_path[-2], walk_path[-1]
                    try:
                        tmp = np.random.choice(self.second_markov_weight_dict[pre_one][curr_one][0], p=self.second_markov_weight_dict[pre_one][curr_one][1])
                        walk_path.append(tmp)
                    except:
                        break
                res.append(walk_path)
        return res

    def run(self, walk_per_vertex, n_workers=4, verbose=50):
        # parallel
        res = Parallel(n_jobs=n_workers, verbose=verbose)(
            delayed(self.node2vec)(num, self.walk_length)
            for num in partition_num(walk_per_vertex, n_workers))
        self.walks_generated = sum(res, [])
