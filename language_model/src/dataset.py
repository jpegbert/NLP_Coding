import re
from language_model.src.processing import build_sentences

'''
加载语料，并加入起始标记<s></s>
'''


def load_dataset(file_path):
	with open(file_path, "r") as f:
		return build_sentences([re.split("\s+", line.rstrip('\n')) for line in f])
