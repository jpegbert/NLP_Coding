from collections import defaultdict


def read_graph_data(data_cate_path, data_link_path):
    vertex = defaultdict(list)
    with open(data_cate_path, 'r+') as f:
        for line in f.readlines():
            node_num, cate_num = line.strip().split(' ')
            vertex[cate_num].append(node_num)

    edge = defaultdict(list)
    with open(data_link_path, 'r+') as f:
        for line in f.readlines():
            node_from, node_to = line.strip().split(' ')
            edge[node_from].append(node_to)
    return vertex, edge


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]
