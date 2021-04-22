import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction, BertForQuestionAnswering
from transformers import BertModel


"""
https://www.cnblogs.com/douzujun/p/13572694.html
"""


model_name = 'bert-base-chinese'
MODEL_PATH = './bert-base-chinese/'

# a. 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
# b. 导入配置文件
model_config = BertConfig.from_pretrained(model_name)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
bert_model = BertModel.from_pretrained(MODEL_PATH, config=model_config)


# encode仅返回input_ids
print(tokenizer.encode('吾儿莫慌'))   # [101, 1434, 1036, 5811, 2707, 102]

# encode_plus返回所有编码信息 input_ids：是单词在词典中的编码 token_type_ids：区分两个句子的编码（上句全为0，下句全为1）
# attention_mask：指定 对哪些词 进行self-Attention操作
sen_code = tokenizer.encode_plus('这个故事没有终点', "正如星空没有彼岸")
# print(sen_code)
# [101, 1434, 1036, 5811, 2707, 102]
# {'input_ids': [101, 6821, 702, 3125, 752, 3766, 3300, 5303, 4157, 102, 3633, 1963, 3215, 4958, 3766, 3300, 2516, 2279, 102],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

# 将input_ids转化回token
print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))
# ['[CLS]', '这', '个', '故', '事', '没', '有', '终', '点', '[SEP]', '正', '如', '星', '空', '没', '有', '彼', '岸', '[SEP]']

# 对编码进行转换，以便输入Tensor
tokens_tensor = torch.tensor([sen_code['input_ids']])  # 添加batch维度并,转换为tensor,torch.Size([1, 19])
segments_tensors = torch.tensor(sen_code['token_type_ids'])  # torch.Size([19])

bert_model.eval()

# 进行编码
with torch.no_grad():
    outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
    encoded_layers = outputs  # outputs类型为tuple

    # Bert最终输出的结果维度为：sequence_output, pooled_output, (hidden_states), (attentions)
    # 以输入序列为19为例：
    # sequence_output：torch.Size([1, 19, 768])
    # 输出序列
    # pooled_output：torch.Size([1, 768])
    # 对输出序列进行pool操作的结果
    # (hidden_states)：tuple, 13 * torch.Size([1, 19, 768])
    # 隐藏层状态（包括Embedding层），取决于 model_config 中的 output_hidden_states
    # (attentions)：tuple, 12 * torch.Size([1, 12, 19, 19])
    # 注意力层，取决于 model_config 中的 output_attentions
    print(encoded_layers[0].shape, encoded_layers[1].shape, encoded_layers[2][0].shape, encoded_layers[3][0].shape)
    # torch.Size([1, 19, 768]) torch.Size([1, 768]) torch.Size([1, 19, 768]) torch.Size([1, 12, 19, 19])



# =====================遮蔽语言模型（Masked Language Model 简称MLM任务）======================
model_name = 'bert-base-chinese'    # 指定需下载的预训练模型参数

# 任务一：遮蔽语言模型
# BERT 在预训练中引入 [CLS] 和 [SEP] 标记句子的 开头和结尾
samples = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]'] # 准备输入模型的语句

tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer_text = [tokenizer.tokenize(i) for i in samples] # 将句子分割成一个个token，即一个个汉字和分隔符
# [['[CLS]', '中', '国', '的', '首', '都', '是', '哪', '里', '？', '[SEP]', '北', '京', '是', '[MASK]', '国', '的', '首', '都', '。', '[SEP]']]
# print(tokenizer_text)

input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer_text]
input_ids = torch.LongTensor(input_ids)
# print(input_ids)
# tensor([[ 101,  704, 1744, 4638, 7674, 6963, 3221, 1525, 7027, 8043,  102, 1266,
#           776, 3221,  103, 1744, 4638, 7674, 6963,  511,  102]])

# 读取预训练模型
model = BertForMaskedLM.from_pretrained(model_name, cache_dir='./Transformer-Bert/')
model.eval()

outputs = model(input_ids)
prediction_scores = outputs[0]                 # prediction_scores.shape=torch.Size([1, 21, 21128])
sample = prediction_scores[0].detach().numpy() # (21, 21128)

# 21为序列长度，pred代表每个位置最大概率的字符索引
pred = np.argmax(sample, axis=1)  # (21,)
# ['，', '中', '国', '的', '首', '都', '是', '哪', '里', '？', '。', '北', '京', '是', '中', '国', '的', '首', '都', '。', '。']
print(tokenizer.convert_ids_to_tokens(pred))
print(tokenizer.convert_ids_to_tokens(pred)[14])  # 被标记的[MASK]是第14个位置, 中


# ==========================句子预测任务(Next Sentence Prediction 简称NSP任务)===============================
# sen_code1 = tokenizer.encode_plus('今天天气怎么样？', '今天天气很好！')
# sen_code2 = tokenizer.encode_plus('明明是我先来的！', '我喜欢吃西瓜！')

# tokens_tensor = torch.tensor([sen_code1['input_ids'], sen_code2['input_ids']])
# print(tokens_tensor)
# tensor([[ 101,  791, 1921, 1921, 3698, 2582,  720, 3416,  102,  791, 1921, 1921,
#          3698, 2523, 1962,  102],
#         [ 101, 3209, 3209, 3221, 2769, 1044, 3341, 4638,  102, 7471, 3449,  679,
#          1963, 1921, 7360,  102]])

# 上面可以换成
samples = ["[CLS]天气真的好啊！[SEP]一起出去玩吧！[SEP]", "[CLS]小明今年几岁了[SEP]我不喜欢学习！[SEP]"]
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenized_text = [tokenizer.tokenize(i) for i in samples]
input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
tokens_tensor = torch.LongTensor(input_ids)


# 读取预训练模型
model = BertForNextSentencePrediction.from_pretrained(model_name, cache_dir='./Transformer-Bert/')
model.eval()

outputs = model(tokens_tensor)
# sequence_output：输出序列
seq_relationship_scores = outputs[0]              # seq_relationship_scores.shape: torch.Size([2, 2])
sample = seq_relationship_scores.detach().numpy() # sample.shape: [2, 2]

pred = np.argmax(sample, axis=1)
print(pred)   # [0 0]， 0表示是上下句关系，1表示不是上下句关系


# =====================问答任务（QA）====================
model_name = 'bert-base-chinese'

# 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
# 导入配置文件
model_config = BertConfig.from_pretrained(model_name)
# 最终有两个输出，初始位置和结束位置
model_config.num_labels = 2

# 根据bert的 model_config 新建 BertForQuestionAnsering
model = BertForQuestionAnswering(model_config)
model.eval()

question, text = '里昂是谁？', '里昂是一个杀手。'

sen_code = tokenizer.encode_plus(question, text)

tokens_tensor = torch.tensor([sen_code['input_ids']])
segments_tensors = torch.tensor([sen_code['token_type_ids']]) # 区分两个句子的编码（上句全为0，下句全为1）

start_pos, end_pos = model(tokens_tensor, token_type_ids=segments_tensors)
# 进行逆编码，得到原始的token
all_tokens = tokenizer.convert_ids_to_tokens(sen_code['input_ids'])
print(all_tokens)  # ['[CLS]', '里', '昂', '是', '谁', '[SEP]', '里', '昂', '是', '一', '个', '杀', '手', '[SEP]']

# 对输出的答案进行解码的过程
answer = ' '.join(all_tokens[torch.argmax(start_pos) : torch.argmax(end_pos) + 1])

# 每次执行的结果不一致，这里因为没有经过微调，所以效果不是很好，输出结果不佳，下面的输出是其中的一种。
print(answer)   # 一 个 杀 手


# =====================BERT一共可以解决四类任务====================

