import torch


class Config(object):
    def __init__(self):
        self.saved_dir = "./output"
        self.train_path = "./dataset/query_wellformedness/train.txt"
        self.test_path = "./dataset/query_wellformedness/test.txt"
        self.pretrained_model_dir = None

        self.embed_dim = 300
        self.hidden_dim = 10

        self.optim = 'sgd'
        self.num_epoch = 100
        self.lr = 0.001
        self.momentum = 0.9
        self.batch_size = 128
        self.valid_interval = 1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class TextRNNConfig(object):
    def __init__(self):
        self.saved_dir = "./output/TextRNN"
        self.train_path = "./dataset/query_wellformedness/train.txt"
        self.test_path = "./dataset/query_wellformedness/test.txt"
        self.pretrained_model_dir = None

        self.embed_dim = 300
        self.hidden_dim = 32
        self.num_rnn_layers = 2
        self.dropout_prob = 0.8

        self.optim = 'sgd'
        self.num_epoch = 100
        self.lr = 0.01
        self.momentum = 0.9
        self.batch_size = 128
        self.valid_interval = 1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
