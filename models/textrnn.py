import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, bidirectional=True, num_labels=1, num_rnn_layers=1,
                 vectors=None, pad_idx=0, dropout_prob=0.5):
        super(TextRNN, self).__init__()
        self.bidirectional = bidirectional
        rnn_hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim

        self.embed_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, rnn_hidden_dim, bidirectional=bidirectional, num_layers=num_rnn_layers,
                            batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc_layer = nn.Linear(hidden_dim*num_rnn_layers, num_labels)
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
