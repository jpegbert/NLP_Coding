
"""
逆向最大匹配算法
"""
class ReverseMaxMatch(object):
    def __init__(self):
        self.window_size = 3
        self.single_word_size = 0 # 单字的个数
        self.cut_size = 0 # 切分的单词数量
        self.name = "逆向最大匹配法"

    def cut(self, text):
        result, index, piece = [], len(text), ""
        self.single_word_size = 0  # 单字的个数
        dic = ["研究", "研究生", "生命", "命", "的", "起源", "南京市", "市长", "长江", "大桥", "长江大桥", "江大桥"]
        while index > 0:
            for size in range(index - self.window_size, index):
                piece = text[size: index]
                if piece in dic:
                    index = size + 1
                    break
            index -= 1
            if len(piece) == 1:
                self.single_word_size += 1
            result.append(str(piece))
        self.cut_size = len(result)
        return result


if __name__ == '__main__':
    text = ["研究生命的起源", "南京市长江大桥"]
    tokenizer = ReverseMaxMatch()
    for sentence in text:
        print(" / ".join(tokenizer.cut(sentence)))
        print("词的数量=%d, 单字的个数=%d" % (tokenizer.cut_size, tokenizer.single_word_size))
