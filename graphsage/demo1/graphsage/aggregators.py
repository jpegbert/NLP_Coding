import torch
import torch.nn as nn
from torch.autograd import Variable
import random

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """
        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        # 如果num_sample设置了具体的数字
        if not num_sample is None:
            _sample = random.sample
            # 首先对每一个节点的邻居集合neigh进行遍历，判断一下已有邻居数和采样数大小，多于采样数进行抽样
            samp_neighs = [_set(_sample(to_neigh, num_sample, )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        # *拆解列表后，转为为多个独立的元素作为参数给union，union函数进行去重合并
        unique_nodes_list = list(set.union(*samp_neighs))
        # 节点标号不一定都是从0开始的，创建一个字典，key为节点ID，value为节点序号
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        # nodes表示batch内的节点，unique_nodes表示batch内的节点用到的所有邻居节点，unique_nodes > nodes
        # 创建一个nodes * unique_nodes大小的矩阵
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        # 遍历每一个邻居集合的每一个元素，并且通过ID(key)获取到节点对应的序号--列切片
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # 行切片，比如samp_neighs = [{3,5,9}, {2,8}, {2}]，行切片为[0,0,0,1,1,2]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # 利用切片创建邻接矩阵
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        # 统计每一个节点的邻居数量
        num_neigh = mask.sum(1, keepdim=True)
        # 分比例
        mask = mask.div(num_neigh)
        # embed_matrix: [n, m]
        # n: unique_nodes
        # m: dim
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        # mean操作
        to_feats = mask.mm(embed_matrix)
        return to_feats


