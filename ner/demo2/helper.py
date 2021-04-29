import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-dd", "--data_dir", type=str, help="train, dev and test data dir", default="data/")
parser.add_argument("-bc", "--bert_config", type=str, help="bert config file dir", default="bert_model/bert_config.json")
parser.add_argument("-ic", "--init_checkpoint", type=str, help="bert model dir", default="bert_model/bert_model.ckpt")
parser.add_argument("-v", "--vocab_dir", type=str, help="vocab dir", default="bert_model/vocab.txt")
parser.add_argument("-e", "--entry", type=str, default="train")
parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5, help="learning rate")
ARGS = parser.parse_args()