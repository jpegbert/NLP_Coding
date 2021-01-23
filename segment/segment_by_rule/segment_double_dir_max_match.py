from segment.segment_by_rule.segment_max_match import MaxMatch
from segment.segment_by_rule.segment_reverse_max_match import ReverseMaxMatch


"""
双向最大匹配算法
"""
class DoubleDirMaxMatch(object):
    def __init__(self):
        # 正向最大匹配和逆向最大匹配
        self.max_match = MaxMatch()
        self.reverse_max_match = ReverseMaxMatch()
        self.name = "双向最大匹配法"
        self.choose_match = ""

    def cut(self, text):
        rnt_max_match = self.max_match.cut(text)
        rnt_reverse_max_match = self.reverse_max_match.cut(text)
        if self.max_match.cut_size != self.reverse_max_match.cut_size:
            # 如果正反向分词结果，词数不同，返回词数较少的分词结果
            if self.max_match.cut_size < self.reverse_max_match.cut_size:
                self.choose_match = self.max_match.name
                return rnt_max_match
            else:
                self.choose_match = self.reverse_max_match.name
                return rnt_reverse_max_match
        else:
            # 如果正反向分词结果，词数相同，返回单字数较少的分词结果
            if self.max_match.single_word_size < self.reverse_max_match.single_word_size:
                self.choose_match = self.max_match.name
                return rnt_max_match
            else: # 如果单字数也相同，就返回最大逆向匹配（实验表明，最大逆向结果准确性比较高）
                self.choose_match = self.reverse_max_match.name
                return rnt_reverse_max_match


if __name__ == '__main__':
    text = ["研究生命的起源", "南京市长江大桥"]
    tokenizer = DoubleDirMaxMatch()
    for sentence in text:
        print(" / ".join(tokenizer.cut(sentence)))
        print("选择的分词算法=%s" % tokenizer.choose_match)
