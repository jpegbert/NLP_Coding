from collections import defaultdict
import numpy as np


def alias_setup(probs):
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	# 将各个概率分成两组，一组概率值大于1，一组概率值小于1
	for kk, prob in enumerate(probs):
		q[kk] = K * prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	# 使用贪心算法，将概率值小于1的不断填满
	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
		# 更新概率值
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)
	return J, q


def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	# 取自己
	if np.random.rand() < q[kk]:
		return kk
	else: # 取alias table存的节点
		return J[kk]


stat = defaultdict(int)

# 抽样1000次，统计每个值出现的频率，验证是否符合按指定概率抽样的结果
for i in range(1000):
    # 对于分别以概率0.1， 0.2， 0.2，0.5出现的事件进行抽样
    J, q = alias_setup([0.1, 0.2, 0.2, 0.5])
    choice = alias_draw(J, q)
    stat[choice] += 1

s = sum(stat.values())
for k in range(0, 4):
    print(k, stat[k] * 1.0 / s)

