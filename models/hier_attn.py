import torch
import torch.nn as nn

from models.attn_birnn import AttentionalBiRNN

class HAN(nn.Module):
    def __init__(self, opts, vectors=None):
        super().__init__()
        self.device = opts.device

        self.emb_lut = nn.Embedding(opts.num_vocab, opts.embedd_dim, opts.pad_idx)
        if vectors:
            self.emb_lut.from_pretrained(embeddings=vectors, freeze=False)
        else:
            nn.init.uniform_(self.emb_lut.weight.data, -opts.init_weight, opts.init_weight)

        self.word_encoder = AttentionalBiRNN(opts.embedd_dim, opts.word_hidden_size,
                                             opts.word_attn_size,
                                             opts.init_weight)

        self.sent_encoder = AttentionalBiRNN(opts.word_hidden_size,
                                             opts.sentence_hidden_size,
                                             opts.sentence_attn_size,
                                             opts.init_weight)

        self.fc = nn.Linear(opts.sentence_hidden_size, opts.num_classes)

    def forward(self, input_ids):
        emb = self.emb_lut(input_ids)


if __name__ == "__main__":
    input_sample = torch.rand((2, 6, 31))