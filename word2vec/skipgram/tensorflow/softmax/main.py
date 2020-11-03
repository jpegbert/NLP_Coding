import argparse
from word2vec.skipgram.tensorflow.softmax.dataset import Corpus, load_data
from word2vec.skipgram.tensorflow.softmax.skipgram import Skipgram

"""
采用tensorflow以skipgram + softmax的方式实现word2vec
"""

parser = argparse.ArgumentParser(description='main.py')
parser.add_argument('-train', action='store_true', default=False, help='train model')
parser.add_argument('-test', action='store_true', default=False, help='test model')
args = parser.parse_args()


if __name__ == '__main__':
	data = list(load_data())
	corpus = Corpus(data)
	skipgram = Skipgram(corpus)

	if args.train:
		skipgram.train()
	elif args.test:
		word = input('Input word> ')
		print(skipgram.test(word))
