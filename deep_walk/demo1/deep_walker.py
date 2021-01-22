import random
from joblib import Parallel, delayed
from deep_walk.demo1.util import read_graph_data, partition_num


class Deep_Walker:
    def __init__(self, data_cate_path, data_link_path, walk_per_vertex,
                 walk_length, n_workers=1, verbose=1):

        self.vertex, self.edge = read_graph_data(data_cate_path, data_link_path)
        self.walk_per_vertex = walk_per_vertex
        self.walk_length = walk_length
        self.vertex_list = sum(list(self.vertex.values()), [])
        self.walks_generated = []

        self.run(walk_length, walk_per_vertex, n_workers, verbose)

    def walk(self, start_vertex, walk_length):
        walk_path = [start_vertex]
        while len(walk_path) < walk_length:
            node_from = walk_path[-1]
            if not self.edge[node_from]:
                break
            walk_path.append(random.choice(self.edge[node_from]))
        return walk_path

    def deep_walk(self, walk_length, walk_per_vertex):
        res = []
        for _ in range(walk_per_vertex):
            random.shuffle(self.vertex_list)
            for v in self.vertex_list:
                one_walk = self.walk(v, walk_length)
                res.append(one_walk)
        return res

    def run(self, walk_length, walk_per_vertex, n_workers=1, verbose=1):
        res = Parallel(n_jobs=n_workers, backend="threading", verbose=verbose)(
            delayed(self.deep_walk)(walk_length, num)
            for num in partition_num(walk_per_vertex, n_workers))
        self.walks_generated = sum(res, [])
        return self.walks_generated

