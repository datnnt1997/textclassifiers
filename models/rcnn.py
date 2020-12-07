import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RCNN(nn.Module):
    def __init__(self, opts, vectors):
        super(RCNN, self).__init__()
        self.bidirectional = opts.bidirectional
        rnn_hidden_dim = opts.hidden_dim // 2 if opts.bidirectional else opts.hidden_dim

        self.embed_layer = nn.Embedding(opts.vocab_size, opts.embed_dim, padding_idx=opts.pad_idx)
        if vectors is not None:
            self.embed_layer.weight.data = nn.Parameter(vectors, requires_grad=True)

        self.lstm = nn.LSTM(opts.embed_dim, rnn_hidden_dim, bidirectional=opts.bidirectional,
                            num_layers=1, batch_first=True, dropout=opts.dropout_prob)

        self.dropout = nn.Dropout(opts.dropout_prob)

        self.fc_layer_1 = nn.Linear(opts.hidden_dim + opts.embed_dim, opts.fc_hidden_dim)

        self.fc_layer_2 = nn.Linear(opts.fc_hidden_dim, opts.num_labels)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, seq_len, label_ids=None):
        token_reps = self.embed_layer(input_ids)
        packed_reps = pack_padded_sequence(token_reps, batch_first=True, lengths=seq_len.tolist())
        packed_output, (hn, cn) = self.lstm(packed_reps)
        lstm_out = pad_packed_sequence(packed_output, batch_first=True)