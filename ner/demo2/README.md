# BERT Chinese NER

github链接 https://github.com/Ma-Dan/BERT-ChineseNER

基于Bi-LSTM + CRF 的中文机构名、人名、地名识别，MSRA NER语料，BIO标注
GOOGLE BERT模型

参考：https://github.com/yanwii/ChineseNER
      https://github.com/macanv/BERT-BiLSTM-CRF-NER

# 下载bert预训练模型
     
    wget -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

放到根目录 **bert_model** 下

# 用法

    # 训练
    python3 model.py -e train
    
    
    # 预测
    python3 model.py -e predict

# 介绍

### bert 模型的加载和使用

    def bert_layer(self):
        # 加载bert配置文件
        bert_config = modeling.BertConfig.from_json_file(ARGS.bert_config)

        # 创建bert模型　
        model = modeling.BertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )
        # 加载词向量
        self.embedded = model.get_sequence_output()
        self.model_inputs = tf.nn.dropout(
            self.embedded, self.dropout
        )

### bert 优化器

    self.train_op = create_optimizer(
        self.loss, self.learning_rate, num_train_steps, num_warmup_steps, False
    )
    



# 例子
    > 金树良先生，董事，硕士。现任北方国际信托股份有限公司总经济师。曾任职于北京大学经济学院国际经济系。1992年7月起历任海南省证券公司副总裁、北京华宇世纪投资有限公司副总裁、昆仑证券有限责任公司总裁、北方国际信托股份有限公司资产管理部总经理及公司总经理助理兼资产管理部总经理、渤海财产保险股份有限公司常务副总经理及总经理、北方国际信托股份有限公司总经理助理。
    >   [
            {
              "begin": 14,
              "end": 26,
              "entity": "北方国际信托股份有限公司",
              "type": "ORG"
            },
            {
              "begin": 70,
              "end": 82,
              "entity": "北京华宇世纪投资有限公司",
              "type": "ORG"
            },
            {
              "begin": 99,
              "end": 111,
              "entity": "北方国际信托股份有限公司",
              "type": "ORG"
            },
            {
              "begin": 160,
              "end": 172,
              "entity": "北方国际信托股份有限公司",
              "type": "ORG"
            },
            {
              "begin": 0,
              "end": 3,
              "entity": "金树良",
              "type": "PER"
            }
        ]