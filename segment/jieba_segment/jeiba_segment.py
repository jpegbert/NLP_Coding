import jieba
import jieba.posseg as posseg
import jieba.analyse as analyse

"""
https://www.jianshu.com/p/2cccb07d9a4e
"""

text = "征战四海只为今日一胜，我不会再败了。"


def basic_demo():
    # jieba.cut直接得到generator形式的分词结果
    seg = jieba.cut(text)
    print(' '.join(seg))

    # 也可以使用jieba.lcut得到list的分词结果
    seg = jieba.lcut(text)
    print(seg)


def cixingfenxi_demo():
    """
    词性分析
    """
    # generator形式形如pair(‘word’, ‘pos’)的结果
    seg = posseg.cut(text)
    print([se for se in seg])

    # list形式的结果
    seg = posseg.lcut(text)
    print(seg)


def keyword_extract():
    """
    关键词抽取
    """
    # TF-IDF
    tf_result = analyse.extract_tags(text, topK=5)  # topK指定数量，默认20
    print(tf_result)
    # TextRank
    tr_result = analyse.textrank(text, topK=5)  # topK指定数量，默认20
    print(tr_result)

    jieba.analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=(), withFlag=False)
    # topK 表示返回最大权重关键词的个数，None表示全部
    # withWeight表示是否返回权重，是的话返回(word,weight)的list
    # allowPOS仅包括指定词性的词，默认为空即不筛选。
    jieba.analyse.textrank(text, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False)
    # 与TF-IDF方法相似，但是注意allowPOS有默认值，即会默认过滤某些词性。


def fenci():
    """
    分词
    """
    text = '今天天气真好'
    text = '电脑组装维护技术员'
    res1 = jieba.cut(text, cut_all=False, HMM=True)  # 精确模式
    res2 = jieba.cut(text, cut_all=True, HMM=True)  # 全模式
    res3 = jieba.cut_for_search(text, HMM=True)  # 搜索引擎模式
    print(' '.join(res1))
    print(' '.join(res2))
    print(' '.join(res3))


def load_dict():
    """
    加载自定义词典
    """
    file_name = ""
    jieba.load_userdict(file_name)  # 载入自定义词典
    word = ""
    jieba.add_word(word, freq=None, tag=None)  # 在程序中动态修改词典
    jieba.del_word(word)
    segment = ""
    jieba.suggest_freq(segment, tune=True)  # 调节单个词语的词频，使其能/不能被分词开


def paralell():
    """
    并行分词
    """
    jieba.enable_parallel(4)  # 开启并行分词模式，参数为并行进程数，默认全部
    jieba.disable_parallel()  # 关闭并行分词模式


def main():
    basic_demo()
    cixingfenxi_demo() # 词性分析
    keyword_extract() # 关键词抽取
    fenci() # 分词
    load_dict() # 加载自定义词典
    paralell() # 并行分词


if __name__ == '__main__':
    main()

