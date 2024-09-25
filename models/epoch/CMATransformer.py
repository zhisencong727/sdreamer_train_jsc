from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.attention import MultiHeadAttention, MultiHeadAttention2
from layers.patchEncoder import LinearPatchEncoder, LinearPatchEncoder2
from layers.norm import PreNorm


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        c_in = 1
        c_out = args.c_out
        d_model = args.d_model
        n_heads = args.n_heads
        seq_len = args.seq_len
        dropout = args.dropout
        e_layers = args.e_layers
        patch_len = args.patch_len

        norm_type = args.norm_type
        activation = args.activation

        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1

        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, inner_dim))

        self.cls_eeg = nn.Parameter(torch.randn(1, 1, inner_dim))
        self.cls_emg = nn.Parameter(torch.randn(1, 1, inner_dim))

        self.patch_encs = nn.ModuleList(
            [
                LinearPatchEncoder2(trace_idx, patch_len, c_in, inner_dim)
                for trace_idx in range(n_traces)
            ]
        )

        self.eeg_transformer = nn.ModuleList(
            [
                MultiHeadAttention2(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                    mult=mult_ff,
                )
                for _ in range(e_layers)
            ]
        )

        self.emg_transformer = nn.ModuleList(
            [
                MultiHeadAttention2(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                    mult=mult_ff,
                )
                for _ in range(e_layers)
            ]
        )

        self.eeg_cma = MultiHeadAttention2(
            inner_dim,
            n_heads,
            d_head,
            context_dim=inner_dim,
            dropout=dropout,
            activation=activation,
            norm=norm_type,
            mult=mult_ff,
        )

        self.emg_cma = MultiHeadAttention2(
            inner_dim,
            n_heads,
            d_head,
            context_dim=inner_dim,
            dropout=dropout,
            activation=activation,
            norm=norm_type,
            mult=mult_ff,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(inner_dim * n_traces), nn.Linear(inner_dim * n_traces, c_out)
        )

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, trace, channel, inner_dim]
        eeg, emg = [patch_enc(x) for patch_enc in self.patch_encs]
        b, n, d = eeg.shape
        eeg = eeg + self.pos_embedding[:, :n]
        emg = emg + self.pos_embedding[:, :n]
        cls_eeg = repeat(self.cls_eeg, "() n d -> b n d", b=b)
        cls_emg = repeat(self.cls_emg, "() n d -> b n d", b=b)
        src_eeg = torch.cat([eeg, cls_eeg], dim=-2)
        src_emg = torch.cat([emg, cls_emg], dim=-2)
        for eeg_block, emg_block in zip(self.eeg_transformer, self.emg_transformer):
            src_eeg = eeg_block(src_eeg, context=None)
            src_emg = emg_block(src_emg, context=None)
        cma_eeg = self.eeg_cma(src_eeg, context=src_emg)
        cma_emg = self.eeg_cma(src_emg, context=src_eeg)

        cls_eeg, cls_emg = cma_eeg[:, -1], cma_emg[:, -1]
        # x_our --> [b, n, 2d]
        emb = torch.cat([cls_eeg, cls_emg], dim=-1)
        out = self.mlp_head(emb)
        return out, None, None, emb, label


class Mono_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        c_in = 1
        c_out = args.c_out
        d_model = args.d_model
        n_heads = args.n_heads
        seq_len = args.seq_len
        dropout = args.dropout
        e_layers = args.e_layers
        patch_len = args.patch_len
        emb_dropout = args.emb_dropout
        norm_type = args.norm_type
        activation = args.activation

        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1
        self.features = args.features
        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, inner_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, inner_dim))

        self.patch_enc = LinearPatchEncoder2(0, patch_len, c_in, inner_dim)

        self.transformer = nn.ModuleList(
            [
                MultiHeadAttention(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                    mult=mult_ff,
                )
                for _ in range(e_layers)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(inner_dim * n_traces), nn.Linear(inner_dim * n_traces, c_out)
        )

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, trace, channel, inner_dim]
        x = x[:, :1] if self.features == "EEG" else x[:, -1:]
        x = self.patch_enc(x)
        b, n, d = x.shape
        x = x + self.pos_embedding[:, :n]
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        src_x = torch.cat([x, cls_tokens], dim=-2)

        for block in self.transformer:
            src_x = block(src_x, context=None)

        # emb --> [b, n, d]
        emb = src_x[:, -1]

        out = self.mlp_head(emb)
        return out, None, None, emb, label
