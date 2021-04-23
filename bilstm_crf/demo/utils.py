from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def load_dataset(path, batch_size=64, pad_len=30):
    dataset = []
    with open(path, encoding="utf8") as f:
        data_batch = []
        for line in f:
            ll = line.strip().split("\t")
            while len(ll) < pad_len:
                ll.append("")
            data_batch.append(ll[:pad_len])
            if len(data_batch) == batch_size:
                dataset.append(data_batch)
                data_batch = []
    return dataset


def load_2id_dic(path):
    dic_get = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ll = line.strip().split("\t")
            if len(ll) < 2:
                dic_get[""] = 0
            else:
                dic_get[ll[0]] = int(ll[1])
    return dic_get


def load_id2_dic(path):
    dic_get = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ll = line.strip().split("\t")
            if len(ll) < 2:
                dic_get.append("")
            else:
                dic_get.append(ll[1])
    return dic_get


def item2id_batch(items_batch, dic_get):
    res = []
    for batch_ in items_batch:
        res_batch = []
        for item in batch_:
            sentence = []
            for i in item:
                if i in dic_get:
                    sentence.append(dic_get[i])
            res_batch.append(sentence)
        res.append(res_batch)
    return res


def model_rep(y_true, y_pred, average="micro"):
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1score = f1_score(y_true, y_pred, average=average)
    return p, r, f1score


def model_conf(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def print_matrix(mat):
    for idx in range(len(mat)):
        for j in mat[idx]:
            print("%s\t" % j, end="")
        print("\n", end="")
    print("",end="\n")

# print_matrix([[1,2,3],[3,4,5]])
