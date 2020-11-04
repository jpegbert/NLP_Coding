import jieba

"""
https://blog.csdn.net/aaalswaaa1/article/details/84074815
基本思路：每个词将自己的分数平均投给附近的词，迭代至收敛或指定次数即可，初始分可以打1
"""


def get_stopword_list():
    path = './data/stop_words.utf8'
    stopword_list = [sw.replace('\n', '') for sw in open(path, 'r', encoding='utf8').readlines()]
    return stopword_list


def seg2list(text):
    return jieba.cut(text)


def word_filter(seg_list):
    stopword_list = get_stopword_list()
    filter_list = []
    for w in seg_list:
        if not w in stopword_list and len(w) > 1:
            filter_list.append(w)
    return filter_list


str = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
      '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
      '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
      '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
      '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
      '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
      '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
      '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
      '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
      '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
      '常委会主任陈健倩介绍了大会的筹备情况。'
win = {}
seg_list = seg2list(str)
filter_list = word_filter(seg_list)
# 构建投分表，根据窗口
for i in range(len(filter_list)):
    if filter_list[i] not in win.keys():
        win[filter_list[i]] = set()
    if i - 5 < 0:
        lindex = 0
    else:
        lindex = i - 5
    for j in filter_list[lindex:i + 5]:
        win[filter_list[i]].add(j)

# 投票
time = 0
score = {w: 1.0 for w in filter_list}
while (time < 50):
    for k, v in win.items():
        s = score[k] / len(v)
        score[k] = 0
        for i in v:
            score[i] += s
    time += 1

l = sorted(score.items(), key=lambda score: score[1], reverse=True)
print(l)

