import torch
import torch.nn as nn

from models.common import init_lstm_


class RNNNet(nn.Module):
    """
    
    """
    def __init__(self, embed_dim, rnn_type, hidden_size, bidirectonal, num_rnn_layer, num_vocab, num_classes,
                 dropout=0.5, pad_idx=0, init_weight=0.1, device="cuda", vectors=None):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.bidirect = bidirectonal
        # Embedding look-table layer.
        self.emb_lut = nn.Embedding(num_vocab, embed_dim, pad_idx)
        if vectors is not None:
            self.emb_lut.from_pretrained(embeddings=vectors, freeze=False, padding_idx=pad_idx)
        else:
            nn.init.uniform_(self.emb_lut.weight.data, -init_weight, init_weight)

        hidden_dim = hidden_size//2 if bidirectonal else hidden_size
        if rnn_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_rnn_layer, bidirectional=True, batch_first=True,
                              dropout=dropout)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_rnn_layer, bidirectional=True, batch_first=True,
                               dropout=dropout)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_rnn_layer, bidirectional=True, batch_first=True,
                              dropout=dropout)
        else:
            raise print(f"Type {rnn_type} is not exist!")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

        # Init RNN's weight
        init_lstm_(self.rnn, init_weight)

    def forward(self, inputs, lengths):
        emb = self.emb_lut(inputs)
        packed_batch = nn.utils.rnn.pack_padded_sequence(emb, lengths=lengths.cpu().tolist(), batch_first=True)
        last_outputs, (last_hidden, last_ouput) = self.rnn(packed_batch)
        if self.bidirect:
            sent_enc = torch.cat([last_hidden[-2, :, :], last_hidden[-1, :, :]], dim=-1)
        else:
            sent_enc = last_hidden[-1, :, :]
        logits = self.fc(sent_enc)
        prods = self.softmax(logits)
        return prods, logits


if __name__ == "__main__":
    sample_inputs = torch.randint(0, 100, (4, 10))
    sample_lengths = torch.tensor([10, 9, 7, 3])
    model = RNNNet(embed_dim=50, rnn_type="lstm", hidden_size=100, bidirectonal=True, num_rnn_layer=1, num_vocab=100,
                   num_classes=4, dropout=0.5, pad_idx=0, init_weight=0.1, device="cuda", vectors=None)
    prods, logits = model(sample_inputs, sample_lengths)
    print(logits)
    print(prods)
