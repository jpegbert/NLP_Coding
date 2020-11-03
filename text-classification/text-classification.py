import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn import svm


"""
https://github.com/jpegbert/NLP/tree/master/文本分类
"""


def one_hot():
    pass


def bags_of_words():
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = CountVectorizer()
    res = vectorizer.fit_transform(corpus).toarray()
    print(res)


def n_gram():
    pass


def tf_idf():
    pass


def count_vectors_ridge_classifier():
    """
    Count Vectors + RidgeClassifier
    """
    train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)
    vectorizer = CountVectorizer(max_features=3000)
    train_test = vectorizer.fit_transform(train_df['text'])

    clf = RidgeClassifier()
    clf.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = clf.predict(train_test[10000:])
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))


def tf_idf__ridge_classifier():
    """
    TF - IDF + RidgeClassifier
    """
    train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
    train_test = tfidf.fit_transform(train_df['text'])

    clf = RidgeClassifier()
    clf.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = clf.predict(train_test[10000:])
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))


def text_classify_compare():
    """
    文本分类方法对比
    """
    count_vectors_ridge_classifier()
    tf_idf__ridge_classifier()


def text_classify_influence_by_add_regularization():
    """
    探究正则化对文本分类的影响
    """
    train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)
    sample = train_df[0:5000]
    n = int(2 * len(sample) / 3)
    tfidf = TfidfVectorizer(ngram_range=(2, 3), max_features=2500)
    train_test = tfidf.fit_transform(sample['text'])
    train_x = train_test[:n]
    train_y = sample['label'].values[:n]
    test_x = train_test[n:]
    test_y = sample['label'].values[n:]

    f1 = []
    for i in range(10):
        clf = RidgeClassifier(alpha=0.15 * (i + 1), solver='sag')
        clf.fit(train_x, train_y)
        val_pred = clf.predict(test_x)
        f1.append(f1_score(test_y, val_pred, average='macro'))

    plt.plot([0.15 * (i + 1) for i in range(10)], f1)
    plt.xlabel('alpha')
    plt.ylabel('f1_score')
    plt.show()


def text_classify_influence_by_max_features():
    """
    max_features对文本分类的影响
    """
    train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)
    sample = train_df[0:5000]
    n = int(2 * len(sample) / 3)
    f1 = []
    features = [1000, 2000, 3000, 4000]
    for i in range(4):
        tfidf = TfidfVectorizer(ngram_range=(2, 3), max_features=features[i])
        train_test = tfidf.fit_transform(sample['text'])
        train_x = train_test[:n]
        train_y = sample['label'].values[:n]
        test_x = train_test[n:]
        test_y = sample['label'].values[n:]
        clf = RidgeClassifier(alpha=0.1 * (i + 1), solver='sag')
        clf.fit(train_x, train_y)
        val_pred = clf.predict(test_x)
        f1.append(f1_score(test_y, val_pred, average='macro'))

    plt.plot(features, f1)
    plt.xlabel('max_features')
    plt.ylabel('f1_score')
    plt.show()


def text_classify_influence_by_ngram_range():
    """
    ngram_range对文本分类的影响
    """
    train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)
    sample = train_df[0:5000]
    n = int(2 * len(sample) / 3)
    f1 = []
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=2000)
    train_test = tfidf.fit_transform(sample['text'])
    train_x = train_test[:n]
    train_y = sample['label'].values[:n]
    test_x = train_test[n:]
    test_y = sample['label'].values[n:]
    clf = RidgeClassifier(alpha=0.1, solver='sag')
    clf.fit(train_x, train_y)
    val_pred = clf.predict(test_x)
    f1.append(f1_score(test_y, val_pred, average='macro'))

    tfidf = TfidfVectorizer(ngram_range=(2, 2), max_features=2000)
    train_test = tfidf.fit_transform(sample['text'])
    train_x = train_test[:n]
    train_y = sample['label'].values[:n]
    test_x = train_test[n:]
    test_y = sample['label'].values[n:]
    clf = RidgeClassifier(alpha=0.1, solver='sag')
    clf.fit(train_x, train_y)
    val_pred = clf.predict(test_x)
    f1.append(f1_score(test_y, val_pred, average='macro'))

    tfidf = TfidfVectorizer(ngram_range=(3, 3), max_features=2000)
    train_test = tfidf.fit_transform(sample['text'])
    train_x = train_test[:n]
    train_y = sample['label'].values[:n]
    test_x = train_test[n:]
    test_y = sample['label'].values[n:]
    clf = RidgeClassifier(alpha=0.1, solver='sag')
    clf.fit(train_x, train_y)
    val_pred = clf.predict(test_x)
    f1.append(f1_score(test_y, val_pred, average='macro'))

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=2000)
    train_test = tfidf.fit_transform(sample['text'])
    train_x = train_test[:n]
    train_y = sample['label'].values[:n]
    test_x = train_test[n:]
    test_y = sample['label'].values[n:]
    clf = RidgeClassifier(alpha=0.1, solver='sag')
    clf.fit(train_x, train_y)
    val_pred = clf.predict(test_x)
    f1.append(f1_score(test_y, val_pred, average='macro'))


def text_classify_influence_by_parameters():
    text_classify_influence_by_add_regularization() # 正则化对文本分类的影响
    text_classify_influence_by_max_features() # max_features对文本分类的影响
    text_classify_influence_by_ngram_range() # ngram_range对文本分类的影响


def tf_idf_logistic_regression():
    """
    TF IDF + Logistic Regression
    """
    train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    train_test = tfidf.fit_transform(train_df['text'])  # 词向量 15000*max_features

    reg = linear_model.LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    reg.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = reg.predict(train_test[10000:])
    print('预测结果中各类新闻数目')
    print(pd.Series(val_pred).value_counts())
    print('\n F1 score为')
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))


def tf_idf_sgd_classifier():
    """
    TF IDF + SGDClassifier
    """
    train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    train_test = tfidf.fit_transform(train_df['text'])  # 词向量 15000*max_features

    reg = linear_model.SGDClassifier(loss="log", penalty='l2', alpha=0.0001, l1_ratio=0.15)
    reg.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = reg.predict(train_test[10000:])
    print('预测结果中各类新闻数目')
    print(pd.Series(val_pred).value_counts())
    print('\n F1 score为')
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))


def tf_idf_svm_classifier():
    """
    SVM分类器
    """
    train_df = pd.read_csv('./data/train_set.csv', sep='\t', nrows=15000)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    train_test = tfidf.fit_transform(train_df['text'])  # 词向量 15000*max_features

    reg = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', decision_function_shape='ovr')
    reg.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = reg.predict(train_test[10000:])
    print('预测结果中各类新闻数目')
    print(pd.Series(val_pred).value_counts())
    print('\n F1 score为')
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))


def other_classify_method():
    """
    探究其他文本分类方法对
    """
    tf_idf_logistic_regression() # Logistic Regression分类器
    tf_idf_sgd_classifier() # SGD分类器
    tf_idf_svm_classifier() # SVM分类器


def main():
    one_hot() # one-hot编码
    bags_of_words() # Bag of Words
    n_gram() # N-gram
    tf_idf() #
    text_classify_compare() # 文本分类方法对比
    text_classify_influence_by_parameters() # 探究参数对文本分类的影响
    other_classify_method() # 探究其他文本分类方法对


if __name__ == '__main__':
    main()
