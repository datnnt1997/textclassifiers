import torch
import torch.nn as nn

from models.common import init_lstm_


class AttentionalBiRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_dim, init_weight):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=(hidden_dim//2),
                          bidirectional=True, batch_first=True)
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.V = nn.Parameter(torch.randn(hidden_dim).float())
        # Init GRU's weight
        init_lstm_(self.gru, init_weight)

    def forward(self, inputs, lengths):
        packed_batch = nn.utils.rnn.pack_padded_sequence(inputs, lengths=lengths.tolist(), batch_first=True)
        last_outputs, _ = self.gru(packed_batch)
        enc_sents, len_s = torch.nn.utils.rnn.pad_packed_sequence(last_outputs)

        alpha = torch.tanh(self.W(enc_sents))
        context = self.V(alpha)
        all_att = self._masked_softmax(context, self._list_to_bytemask(list(len_s))).transpose(0, 1)
