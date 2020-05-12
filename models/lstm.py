
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.nn import Parameter


class LSTM(nn.Module):
    def __init__(self, config: dict, vectors: torch.Tensor = None):
        super().__init__()
        target_class = config['target_class']
        self.is_bidirectional = config['bidirectional']
        self.has_bottleneck_layer = config['bottleneck_layer']
        self.tar = config['tar']
        self.ar = config['ar']
        self.device = config['device']
        self.beta_ema = config['beta_ema']  # Temporal averaging
        self.wdrop = config['wdrop']  # Weight dropping
        self.embed_droprate = config['embed_droprate']  # Embedding dropout

        if config['mode'] == 'rand':
            rand_embed_init = torch.Tensor(config['words_num'], config['embed_dim']).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
            self.embed.padding_idx = config['pad_idx']
        elif config['mode'] == 'static':
            self.embed = nn.Embedding.from_pretrained(vectors, freeze=True)
        elif config['mode'] == 'non-static':
            self.embed = nn.Embedding.from_pretrained(vectors, freeze=False)
        else:
            print("Unsupported Mode")
            exit()

        self.lstm = nn.LSTM(config['embed_dim'], config['hidden_dim'], dropout=config['dropout'],
                            num_layers=config['num_layers'], bidirectional=self.is_bidirectional, batch_first=True)

        self.fc1 = nn.Linear(2 * config['hidden_dim'], target_class)
        self.dropout = nn.Dropout(config['dropout'])
        self.softmax = nn.Softmax(dim=-1)
        if config['device'] == 'cuda':
            self = self.cuda()

    def forward(self, x, lengths=None):
        x = self.embed(x)
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        rnn_outs, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        #rnn_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=True)
        x = self.dropout(hidden)
        scores = self.fc1(x)
        return self.softmax(scores), scores
