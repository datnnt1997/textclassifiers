import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super(PositionalEncoding, self).__init__()
        lookup_table = np.array([[pos / np.power(10000, 2 * (i // 2) / embed_dim) for i in range(embed_dim)]
                        for pos in range(max_len)])  # pos/10000^(2i/embed_dim)
        lookup_table[:, 0::2] = np.sin(lookup_table[:, 0::2])  # since(2i)
        lookup_table[:, 1::2] = np.cos(lookup_table[:, 1::2])  # cosine(2i+1)
        
        self.register_buffer('pe', torch.FloatTensor(lookup_table).unsqueeze(0))

    def forward(self, seq_len):
        return self.pe[:, :seq_len].clone().detach()


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx, max_len, dropout_prob):
        super(TransformerEmbedding, self).__init__()
        self.token_embed_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed_layer = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        token_emb = self.token_embed_layer(input_ids)
        pos_enc = self.pos_embed_layer(seq_len)
        token_reps = self.dropout(token_emb + pos_enc)
        return token_reps


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob):
        super(MultiHeadAttention, self).__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=False)
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask=None):
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, 2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs, value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        attn_hidden_states = self.dropout(self.dense(context_layer))
        attn_hidden_states += hidden_states
        attn_hidden_states = self.layer_norm(attn_hidden_states)
        return attn_hidden_states


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super(PositionwiseFeedForward, self).__init__()
        self.first_fc_layer = nn.Linear(hidden_size, intermediate_size)
        self.second_fc_layer = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, atten_hidden_states):
        hidden_states = F.relu(self.first_fc_layer(atten_hidden_states))
        hidden_states = self.second_fc_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += atten_hidden_states
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention_layer = MultiHeadAttention(hidden_size, num_heads, dropout_prob)
        self.pos_feedforward_layer = PositionwiseFeedForward(hidden_size, intermediate_size, dropout_prob)

    def forward(self, hidden_states, attn_mask=None):
        attn_hidden_states = self.multi_head_attention_layer(hidden_states, attn_mask)
        output_hidden_states = self.pos_feedforward_layer(attn_hidden_states)
        return output_hidden_states


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_heads, num_layers, pad_idx, max_len, dropout_prob):
        super(TransformerEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.embed_layer = TransformerEmbedding(vocab_size, hidden_size, pad_idx, max_len, dropout_prob)
        self.encoder_stack = nn.ModuleList([
            EncoderLayer(hidden_size, intermediate_size, num_heads, dropout_prob) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def get_padding_mask(self, input_ids):
        mask = (input_ids != self.pad_idx)
        return mask.unsqueeze(-2)

    def forward(self, input_ids):
        seq_mask = self.get_padding_mask(input_ids)
        hidden_states = self.embed_layer(input_ids)
        hidden_states = self.layer_norm(hidden_states)
        for encoder_layer in self.encoder_stack:
            hidden_states = encoder_layer(hidden_states, seq_mask)
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, opts, vectors):
        super(Transformer, self).__init__()
        self.transformer_encoder_layer = TransformerEncoder(opts.vocab_size, opts.hidden_dim, opts.inter_dim,
                                                            opts.num_heads, opts.num_layers, opts.pad_idx, opts.max_len,
                                                            opts.dropout_prob)
        self.dropout = nn.Dropout(opts.dropout_prob)
        self.fc_layer = nn.Linear(opts.hidden_dim * opts.max_len, opts.num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, seq_len, label_ids=None):
        hidden_state = self.transformer_encoder_layer(input_ids)
        hidden = self.dropout(hidden_state)
        logits = self.fc_layer(hidden.view(hidden.shape[0], -1))
        probs = self.softmax(logits)
        if label_ids is None:
            return None, probs
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label_ids)
            return loss, probs
