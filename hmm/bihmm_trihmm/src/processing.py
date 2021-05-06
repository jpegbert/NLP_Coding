from hmm.bihmm_trihmm.src import const

"""
句子的处理，字典的构建
"""


def build_tags(tags):
	"""
	加入起始标记
	:param tags:
	:return:
	"""
	out = []
	for sentence in tags:
		items = [x.lower() for x in sentence]
		items.insert(0, const.START_TOKEN)
		items.append(const.END_TOKEN)
		out.append(items)
	return out


def build_undict(tags):
	"""
	构建ungram词频词典
	:param tags:
	:return:
	"""
	undict = {}
	for items in tags:
		for word in items:
			if word == const.START_TOKEN or word == const.END_TOKEN:
				continue
			if word not in undict:
				undict[word] = 1
			else:
				undict[word] += 1
	return undict


def build_bidict(tags):
	"""
	构建bigram词频词典，其中以三元组(u, v)作为词典的键
	:param tags:
	:return:
	"""
	bidict = {}
	for items in tags: 
		for i in range(len(items)-1):
			tup = (items[i], items[i+1])
			if tup not in bidict:
				bidict[tup] = 1
			else:
				bidict[tup] += 1
	return bidict


def build_tridict(tags):
	"""
	构建trigram词频词典，其中以三元组(u, v, w)作为词典的键
	:param tags:
	:return:
	"""
	tridict = {}
	for items in tags:
		items.insert(0, const.START_TOKEN)
		for i in range(len(items) -2):
			tup = (items[i], items[i+1], items[i+2])
			if tup not in tridict:
				tridict[tup] = 1
			else:
				tridict[tup] += 1
	return tridict


def build_count_dict(datas, tags):
	"""
	构建(词,词性)词频字典，以及统计词频
	:param datas:
	:param tags:
	:return:
	"""
	tagword_dict = {}
	wordcount = {}
	tagcount = {}
	for i, data in enumerate(datas):
		tag = tags[i][1:-1]
		for idx, d in enumerate(data):
			tup = (tag[idx], d)
			if tup not in tagword_dict:
				tagword_dict[tup] = 1
			else:
				tagword_dict[tup] += 1

			if d not in wordcount:
				wordcount[d] = 1
			else:
				wordcount[d] += 1
			if tag[idx] not in tagcount:
				tagcount[tag[idx]] = 1
			else:
				tagcount[tag[idx]] += 1
	return tagword_dict, wordcount, tagcount
