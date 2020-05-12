
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.nn import Parameter

nn.Transformer

def embedded_dropout(embed, words, dropout=0.1, scale=None, device='cuda'):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    emb = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type,
                                        embed.scale_grad_by_freq, embed.sparse)
    return emb


class WeightDrop(torch.nn.Module):

    def __init__(self, module, weights, dropout=0, variational=False):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def null_function(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.null_function

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class RegLSTM(nn.Module):
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
        if config['device'] == 'cuda':
            self.lstm = self.lstm.cuda()
        if self.wdrop:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=self.wdrop)
        self.dropout = nn.Dropout(config['dropout'])
        self.softmax = nn.Softmax(dim=-1)
        if self.has_bottleneck_layer:
            if self.is_bidirectional:
                self.fc1 = nn.Linear(2 * config['hidden_dim'], config['hidden_dim'])  # Hidden Bottleneck Layer
                self.fc2 = nn.Linear(config['hidden_dim'], target_class)
            else:
                self.fc1 = nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2)  # Hidden Bottleneck Layer
                self.fc2 = nn.Linear(config['hidden_dim'] // 2, target_class)
        else:
            if self.is_bidirectional:
                self.fc1 = nn.Linear(2 * config['hidden_dim'], target_class)
            else:
                self.fc1 = nn.Linear(config['hidden_dim'], target_class)

        if self.beta_ema > 0:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if config['device'] == 'cuda':
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

        if config['device'] == 'cuda':
            self.embed = self.embed.cuda()
            self.fc1 = self.fc1.cuda()
            self.dropout = self.dropout.cuda()
            self.softmax = self.softmax.cuda()
            if hasattr(self, 'fc2'):
                self.fc2 = self.fc2.cuda()

    def forward(self, x, lengths=None):
        x = embedded_dropout(self.embed, x, dropout=self.embed_droprate if self.training else 0, device=self.device) \
            if self.embed_droprate else self.embed(x)
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        rnn_outs, _ = self.lstm(x)
        rnn_outs_temp = rnn_outs

        if lengths is not None:
            rnn_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=True)
            rnn_outs_temp, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs_temp, batch_first=True)

        x = F.relu(torch.transpose(rnn_outs_temp, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        if self.has_bottleneck_layer:
            x = F.relu(self.fc1(x))
            # x = self.dropout(x)
            if self.tar or self.ar:
                scores = self.fc2(x)
                return self.softmax(scores), scores, rnn_outs.permute(1, 0, 2)
            scores = self.fc2(x)
            return self.softmax(scores), scores
        else:
            if self.tar or self.ar:
                scores = self.fc1(x)
                return self.softmax(scores), scores, rnn_outs.permute(1, 0, 2)
            scores = self.fc1(x)
            return self.softmax(scores), scores

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params