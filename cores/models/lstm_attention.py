import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMAttention(nn.Module):
    def __init__(self, opts, vectors):
        super(LSTMAttention, self).__init__()
        # Embedding layer
        self.bidirectional = opts.bidirectional
        rnn_hidden_dim = opts.hidden_dim // 2 if opts.bidirectional else opts.hidden_dim
        self.embed_layer = nn.Embedding(opts.vocab_size, opts.embed_dim, padding_idx=opts.pad_idx)
        if vectors is not None:
            self.embed_layer.weight.data = nn.Parameter(vectors, requires_grad=True)

        # LSTM layer
        self.lstm = nn.LSTM(opts.embed_dim, rnn_hidden_dim, bidirectional=opts.bidirectional,
                            num_layers=opts.num_rnn_layers, batch_first=True, dropout=opts.dropout_prob)
        self.dropout = nn.Dropout(opts.dropout_prob)

        # Output layer
        self.fc_layer = nn.Linear(opts.hidden_dim * opts.num_rnn_layers, opts.num_labels)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def __get_attetntion_vector(lstm_output, last_hidden):
        seq_rep = last_hidden.squeeze(0)
        atten_weights = torch.bmm(lstm_output, seq_rep.unsqueeze(-1))
        alpha = F.softmax(atten_weights, dim=1)
        return alpha.permute(0, 2, 1)

    def forward(self, input_ids, seq_len, label_ids=None):
        token_reps = self.embed_layer(input_ids)
        packed_reps = pack_padded_sequence(token_reps, batch_first=True, lengths=seq_len.tolist())
        packed_output, (hn, cn) = self.lstm(packed_reps)
        last_hidden = self.dropout(hn)

        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            last_hidden = torch.cat([last_hidden[i, :, :] for i in range(last_hidden.shape[0])], dim=-1)

        alpha_vector = self.__get_attetntion_vector(lstm_output, last_hidden)

        attn_hidden = torch.bmm(alpha_vector, lstm_output)

        logits = self.fc_layer(attn_hidden.squeeze(1))
        probs = self.softmax(logits)

        if label_ids is None:
            return None, probs
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label_ids)
            return loss, probs
