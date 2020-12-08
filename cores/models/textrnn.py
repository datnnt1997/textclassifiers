import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence


class TextRNN(nn.Module):
    def __init__(self, opts, vectors):
        super(TextRNN, self).__init__()
        self.bidirectional = opts.bidirectional
        rnn_hidden_dim = opts.hidden_dim // 2 if opts.bidirectional else opts.hidden_dim

        self.embed_layer = nn.Embedding(opts.vocab_size, opts.embed_dim, padding_idx=opts.pad_idx)

        if vectors is not None:
            self.embed_layer.weight.data = nn.Parameter(vectors, requires_grad=True)

        self.lstm = nn.LSTM(opts.embed_dim, rnn_hidden_dim, bidirectional=opts.bidirectional,
                            num_layers=opts.num_rnn_layers, batch_first=True, dropout=opts.dropout_prob)

        self.dropout = nn.Dropout(opts.dropout_prob)
        self.fc_layer = nn.Linear(opts.hidden_dim * opts.num_rnn_layers, opts.num_labels)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, seq_len, label_ids=None):
        token_reps = self.embed_layer(input_ids)
        packed_reps = pack_padded_sequence(token_reps, batch_first=True, lengths=seq_len.tolist())
        packed_output, (hn, cn) = self.lstm(packed_reps)
        hidden = self.dropout(hn)

        if self.bidirectional:
            hidden = torch.cat([hidden[i, :, :] for i in range(hidden.shape[0])], dim=-1)
        logits = self.fc_layer(hidden)
        probs = self.softmax(logits)

        if label_ids is None:
            return None, probs
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label_ids)
            return loss, probs
