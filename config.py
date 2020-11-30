import torch


class Config(object):
    def __init__(self):
        self.saved_dir = "./output"
        self.train_path = "./dataset/query_wellformedness/train.txt"
        self.test_path = "./dataset/query_wellformedness/train.txt"
        self.pretrained_model_dir = None

        self.embed_dim = 300
        self.hidden_dim = 10

        self.optim = 'sgd'
        self.num_epoch = 4
        self.lr = 0.5
        self.momentum = 0.9
        self.batch_size = 128
        self.valid_interval = 1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"