import argparse
from word2vec.cbow.tensorflow.softmax.dataset import Corpus, load_data
from word2vec.cbow.tensorflow.softmax.cbow import Cbow


"""
采用tensorflow以cbow + softmax的方式实现word2vec
"""


parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('-train', action='store_true', default=False, help='train model')
parser.add_argument('-test', action='store_true', default=False, help='test model')
args = parser.parse_args()


if __name__ == '__main__':
	data = list(load_data())
	corpus = Corpus(data)
	cbow = Cbow(corpus)

	if args.train:
		cbow.train()
	elif args.test:
		word = input('Input word> ')
		print(cbow.test(word))
