import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, opts, vectors):
        super(FastText, self).__init__()
        self.embed_layer = nn.Embedding(opts.vocab_size, opts.embed_dim, padding_idx=opts.pad_idx)
        self.fc1_layer = nn.Linear(opts.embed_dim, opts.num_labels)
        # self.fc2_layer = nn.Linear(opts.hidden_dim, opts.num_labels)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, seq_len, label_ids=None):
        token_reps = self.embed_layer(input_ids)
        seq_reps = token_reps.mean(1)
        logits = self.fc1_layer(seq_reps)
        # logits = self.fc2_layer(hiddens)
        probs = self.softmax(logits)
        if label_ids is None:
            return None, probs
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label_ids)
            return loss, probs
