import torch
from transformers import BertTokenizer, BertModel
# BertTokenizer是分词器
# BertModel是bert模型


"""
七月在线 bert模型深度修炼指南
https://www.julyedu.com/video/play/264/8448
"""


# bert-base-chinese中文版的bert
# bert-base-uncased英文版的bert，不区分大小写
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# return_tensors="pt" 表示返回pytorch的tensor
inputs = tokenizer("我们来试试BERT模型吧", return_tensors="pt")
print(inputs) # inputs包含三个内容，一个是input_ids, 一个是token_type_ids,一个是attention_mask

tokenizer.decode(inputs["input_ids"].data.cpu().numpy().reshape(-1)) # 解码

outputs = model(**inputs)
# sequence_outputs是token embedding，即bert词向量,
# pooled_outputs是sentence embedding，即句子向量
sequence_outputs, pooled_outputs = outputs
print(sequence_outputs.shape) # (batch_size, seq_len, emb_dim)
print(pooled_outputs.shape) # (batch_size, emb_dim)


# 微调 就是把学习率设置的很低，这样模型就不会有太大的变化


class ClassificationModel(torch.nn.Module):
    def __init__(self, config, hidden_size):
        self.bert = BertModel(config)
        self.linear = torch.nn.Linear(hidden_size, 2)

    def forward(self, inputs):
        _, pooled_outputs = self.bert(inputs)
        self.linear(pooled_outputs)

