from language_model.src import const

'''
句子的处理，字典的构建
'''


def build_sentences(sentences):
	"""
	加入起始标记
	:param sentences:
	:return:
	"""
	out = []
	for sentence in sentences:
		words = [x.lower() for x in sentence]
		words.insert(0, const.START_TOKEN)
		words.append(const.END_TOKEN)
		out.append(words)
	return out


def build_undict(sentences):
	"""
	构建ungram词频词典
	:param sentences:
	:return:
	"""
	undict = {}
	total = 0
	for words in sentences:
		for word in words:
			if word not in undict:
				undict[word] = 1
			else:
				undict[word] += 1
			if word != const.START_TOKEN and word != const.END_TOKEN:
				total += 1
	return undict, total


def build_bidict(sentences):
	"""
	构建bigram词频词典，其中以三元组(u, v)作为词典的键
	:param sentences:
	:return:
	"""
	bidict = {}
	for words in sentences: 
		for i in range(len(words)-1):
			tup = (words[i], words[i+1])
			if tup not in bidict:
				bidict[tup] = 1
			else:
				bidict[tup] += 1
	return bidict


def build_tridict(sentences):
	"""
	构建trigram词频词典，其中以三元组(u, v, w)作为词典的键
	:param sentences:
	:return:
	"""
	tridict = {}
	sentences.insert(0, const.START_TOKEN)
	for words in sentences:
		for i in range(len(words) -2):
			tup = (words[i], words[i+1], words[i+2])
			if tup not in tridict:
				tridict[tup] = 1
			else:
				tridict[tup] += 1
	return tridict
