import re
from sklearn.model_selection import train_test_split

SOUTCE_DATA = "./data/renmin.txt"
SOURCE_2_DATA = "./data/renmin_2.txt"
SOURCE_3_DATA = "./data/renmin_3.txt"
SOURCE_4_DATA = "./data/renmin_4.txt"

# 初步提取信息
fout = open(SOURCE_2_DATA, "w", encoding="utf-8")
with open(SOUTCE_DATA, "r", encoding="utf-8") as f:
    for line in f:
        line = line.split('  ')
        i = 1
        while i < len(line) - 1:
            if line[i][0] == '[':
                fout.write(line[i].split('/')[0][1:])
                i += 1
                while i < len(line) - 1 and line[i].find(']') == -1:
                    if line[i] != '':
                        fout.write(line[i].split('/')[0])
                    i += 1
                fout.write(line[i].split('/')[0].strip() + '/' +
                           line[i].split('/')[1][-2:] + ' ')
            elif line[i].split('/')[1] == 'nr':
                word = line[i].split('/')[0]
                i += 1
                if i < len(line) - 1 and line[i].split('/')[1] == 'nr':
                    fout.write(word + line[i].split('/')[0] + '/nr ')
                else:
                    fout.write(word + '/nr ')
                    continue
            else:
                fout.write(line[i] + ' ')
            i += 1
        fout.write('\n')
fout.close()

# 只保留nr、ns和nt
fout = open(SOURCE_3_DATA, "w", encoding="utf-8")
with open(SOURCE_2_DATA, "r", encoding="utf-8") as f:
    for line in f:
        line = line.split(' ')
        i = 0
        while i < len(line)-1:
            if line[i] == '':
                i += 1
                continue
            word = line[i].split('/')[0]
            tag = line[i].split('/')[1]
            if tag == 'nr' or tag == 'ns' or tag == 'nt':
                fout.write(word[0] + "/B_" + tag + " ")
                for j in word[1:len(word)-1]:
                    if j != ' ':
                        fout.write(j + "/M_" + tag + " ")
                fout.write(word[-1] + "/E_" + tag + " ")
            else:
                for wor in word:
                    fout.write(wor+'/O ')
            i += 1
        fout.write('\n')
fout.close()

# 删除标点符号，断句
fout = open(SOURCE_4_DATA, "w", encoding="utf-8")
with open(SOURCE_3_DATA, "r", encoding="utf-8") as f:
    texts = f.read()
    sentences = re.split('[，。！？、‘’“”:]/[O]', texts)
    for sentence in sentences:
        if sentence != " ":
            fout.write(sentence.strip()+'\n')
fout.close()

# -----------------------------------------------------------------
# 数据集最终构建
datas = []
labels = []
linedata = []
linelabel = []
tags = {}
tags[''] = 0
tag_id_tmp = 1
words = {}
words["unk_"] = 0
word_id_tmp = 1
f = open(SOURCE_4_DATA, "r", encoding="utf-8")
for line in f:
    line = line.split()
    linedata = []
    linelabel = []
    numNotO = 0
    for word in line:
        word = word.split('/')
        linedata.append(word[0])
        linelabel.append(word[1])
        if word[0] not in words:
            words[word[0]] = word_id_tmp
            word_id_tmp = word_id_tmp + 1
        # words.add(word[0])
        if word[1] not in tags:
            tags[word[1]] = tag_id_tmp
            tag_id_tmp = tag_id_tmp + 1
        # tags.add(word[1])
        if word[1] != 'O':
            numNotO += 1
    if numNotO != 0:
        datas.append(linedata)
        labels.append(linelabel)
words[""] = word_id_tmp
f.close()

# word&id
fout_w2id = open("./data/word2id_dict", "w", encoding="utf-8")
fout_id2w = open("./data/id2word_dict", "w", encoding="utf-8")
for word_key in words.keys():
    fout_w2id.write("%s\t%s\n" % (word_key, words[word_key]))
    fout_id2w.write("%s\t%s\n" % (words[word_key], word_key))
fout_w2id.close()
fout_id2w.close()

# tag&id
fout_t2id = open("./data/tag2id_dict", "w", encoding="utf-8")
fout_id2t = open("./data/id2tag_dict", "w", encoding="utf-8")
for tag_key in tags.keys():
    fout_t2id.write("%s\t%s\n" % (tag_key, tags[tag_key]))
    fout_id2t.write("%s\t%s\n" % (tags[tag_key], tag_key))
fout_t2id.close()
fout_id2t.close()

x_train, x_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=43)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.25, random_state=43)

with open("./data/x_train", "w", encoding="utf-8") as f:
    for idx in range(len(x_train)):
        write_str = "%s\n" % ("\t".join([str(i) for i in x_train[idx]]))
        f.write(write_str)
with open("./data/x_test", "w", encoding="utf-8") as f:
    for idx in range(len(x_test)):
        write_str = "%s\n" % ("\t".join([str(i) for i in x_test[idx]]))
        f.write(write_str)
with open("./data/x_valid", "w", encoding="utf-8") as f:
    for idx in range(len(x_valid)):
        write_str = "%s\n" % ("\t".join([str(i) for i in x_valid[idx]]))
        f.write(write_str)
with open("./data/y_train", "w", encoding="utf-8") as f:
    for idx in range(len(y_train)):
        write_str = "%s\n" % ("\t".join([str(i) for i in y_train[idx]]))
        f.write(write_str)
with open("./data/y_test", "w", encoding="utf-8") as f:
    for idx in range(len(y_test)):
        write_str = "%s\n" % ("\t".join([str(i) for i in y_test[idx]]))
        f.write(write_str)
with open("./data/y_valid", "w", encoding="utf-8") as f:
    for idx in range(len(y_valid)):
        write_str = "%s\n" % ("\t".join([str(i) for i in y_valid[idx]]))
        f.write(write_str)
