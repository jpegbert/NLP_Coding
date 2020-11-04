from gensim import corpora
from gensim import models
from collections import defaultdict
import math
import functools


class TopicModel(object):
    def __init__(self, doc_list, keyword_num, model='LDA', num_topics=4):
        """
        doc_list：加载数据集方法的返回结果
        keyword_num：关键词数量
        model：主题模型的具体算法
        num_topics：主题模型的主题数量
        """
        # 使用gensim的接口，将文本转换为向量化的表示
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据TF-IDF进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num
        self.num_topics = num_topics
        self.model = self.train_lda()

        # 得到数据集的 主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, num_topics=self.num_topics, id2word=self.dictionary)
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

    def calsim(l1, l2):
        """
        余弦相似度计算
        """
        a, b, c = 0.0, 0.0, 0.0
        for t1, t2 in zip(l1, l2):
            x1 = t1[1]
            x2 = t2[1]
            a += x1 * x1
            b += x1 * x1
            c += x2 * x2
        sim = a / math.sqrt(b * c) if not (b * c) == 0 else 0.0
        return sim

    def topic_extract(self, word_list):
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = self.calsim(v, senttopic)
            sim_dic[k] = sim
            for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
                print(k + "/", end='')
            print()

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary = list(set(dictionary))
        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list
