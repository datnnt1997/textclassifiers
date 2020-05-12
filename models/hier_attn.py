import torch
import torch.nn as nn


class HAN(nn.Module):
    def __init__(self, opts, vectors=None):
        super().__init__()
        self.device = opts.device

        self.emb_lut = nn.Embedding(opts.num_vocab, opts.embedd_dim, opts.pad_idx)
        if vectors:
            self.emb_lut.from_pretrained(embeddings=vectors, freeze=False)
        else:
            nn.init.uniform_(self.emb_lut.weight.data, -opts.init_weight, opts.init_weight)

        self.word_encoder = None
        self.sent_encoder = None

        self.fc = nn.Linear(opts.sentence_hidden_size, opts.num_classes)

    def forward(self, input):
        NotImplementedError