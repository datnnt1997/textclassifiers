import torch
import torch.nn as nn

from torch.nn import functional as F


class TextCNN(nn.Module):
    def __init__(self, opts, vectors):
        super(TextCNN, self).__init__()
        self.num_channel = 1
        self.embed_layer_1 = nn.Embedding(opts.vocab_size, opts.embed_dim, padding_idx=opts.pad_idx)
        if opts.multi_channel:
            self.num_channel = 2
            self.embed_layer_2 = nn.Embedding(opts.vocab_size, opts.embed_dim, padding_idx=opts.pad_idx)

        self.conv_layers = nn.ModuleList([
                nn.Conv2d(in_channels=self.num_channel, out_channels=opts.num_filter,
                          kernel_size=(filter_size, opts.embed_dim)) for filter_size in opts.filter_sizes
        ])

        self.dropout = nn.Dropout(opts.dropout_prob)
        self.fc_layer = nn.Linear(len(opts.filter_sizes) * opts.num_filter, opts.num_labels)
        self.softmax = nn.LogSoftmax(dim=-1)

    @staticmethod
    def conv_block(input_tensor, conv_layer):
        feature_map = conv_layer(input_tensor)
        activated_map = F.relu(feature_map.squeeze(3))
        pooled_map = F.max_pool1d(activated_map, activated_map.size()[2]).squeeze(2)
        return pooled_map

    def forward(self, input_ids, seq_len, label_ids=None):
        token_reps = self.embed_layer_1(input_ids)
        if self.num_channel > 1:
            second_token_reps = self.embed_layer_2(input_ids)
            token_reps = torch.stack([token_reps, second_token_reps], dim=1)

        pooled_maps = [self.conv_block(token_reps, conv) for conv in self.conv_layers]
        hidden = torch.cat(pooled_maps, 1)
        hidden = self.dropout(hidden)
        logits = self.fc_layer(hidden)
        probs = self.softmax(logits)
        if label_ids is None:
            return None, probs
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label_ids)
            return loss, probs


