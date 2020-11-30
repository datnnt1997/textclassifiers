import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, vectors=None, pad_idx=0):
        super(FastText, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1_layer = nn.Linear(embed_dim, hidden_dim)
        self.fc2_layer = nn.Linear(hidden_dim, num_labels)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, seq_len, label_ids=None):
        token_reps = self.embed_layer(input_ids)
        seq_reps = token_reps.mean(1)
        hiddens = self.fc1_layer(seq_reps)
        logits = self.fc2_layer(hiddens)
        probs = self.softmax(logits)
        if label_ids is None:
            return None, probs
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label_ids)
            return loss, probs
