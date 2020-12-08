import torch


class Config(object):
    def __init__(self):
        self.saved_dir = "./output"
        self.train_path = "./dataset/query_wellformedness/train.txt"
        self.test_path = "./dataset/query_wellformedness/test.txt"
        self.data_format = ['text', 'label']
        self.delimiter = '\t'
        self.pretrained_embedding = None
        self.pretrained_model_dir = None

        self.max_len = 256
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_idx = 0
        self.embed_dim = 300
        self.hidden_dim = 10
        self.optim = 'sgd'

        self.decay_rate = 0.9
        self.decay_steps = -1  # if = -1 update after each epoch
        self.num_epoch = 100
        self.lr = 0.001
        self.momentum = 0.9
        self.batch_size = 128
        self.valid_interval = 1
        self.random_seed = 64

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def add_attribute(self, attr_dict):
        for key, value in attr_dict.items():
            if key in self.__dict__:
                self.__dict__[key] = value


class FastTextConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_type = 'fasttext'


class TextRNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.num_rnn_layers = 2
        self.dropout_prob = 0.8
        self.bidirectional = True

        self.model_type = 'textrnn'


class TextCNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.num_filter = 100
        self.filter_sizes = [3, 4, 5]
        self.dropout_prob = 0.2
        self.multi_channel = True

        self.model_type = 'textcnn'


class RCNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.dropout_prob = 0.8
        self.fc_hidden_dim = 64
        self.bidirectional = True

        self.model_type = 'rcnn'


class LSTMAttConfig(Config):
    def __init__(self):
        super().__init__()
        self.num_rnn_layers = 2
        self.dropout_prob = 0.8
        self.bidirectional = True

        self.model_type = 'lstmatt'


class TransformerConfig(Config):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 512
        self.inter_dim = 2048
        self.num_layers = 6
        self.num_heads = 8
        self.dropout_prob = 0.1

        self.model_type = 'transformer'


CONFIG_MAP = {
    'fasttext': FastTextConfig,
    'textrnn': TextRNNConfig,
    'textcnn': TextCNNConfig,
    'rcnn': RCNNConfig,
    'lstmattn': LSTMAttConfig,
    'transformer': TransformerConfig
}

if __name__ == '__main__':
    config = FastTextConfig()
    print()
