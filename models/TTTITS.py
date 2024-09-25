import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.residual_block import EncoderLayer, PositionalEncoding
from layers.last_fusion import (WithoutFusion, V_DAB, ChannelAttention,
                                GroupConv, STAR, ShuffleConv,
                                TSConv2d, TSDeformConv2d)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.seq_len = configs.seq_len
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)
        self.n_heads = configs.n_heads
        actual_c_out = self.c_out * 2
        self.d_lower = configs.d_lower
        self.dropout = configs.dropout

        self.n_groups = 1
        self.n_group_inner_layers = 1

        self.param_sharing_strategy = configs.param_sharing_strategy

        if self.param_sharing_strategy == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(configs)
                    for _ in range(self.n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(configs)
                    for _ in range(self.n_group_inner_layers)
                ]
            )
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(configs)
                    for _ in range(self.n_groups)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(configs)
                    for _ in range(self.n_groups)
                ]
            )

        self.dropout = nn.Dropout(p=self.dropout)
        self.position_enc = PositionalEncoding(self.d_ff, n_position=self.seq_len)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.layer_stack_for_first_block.to(device=device)
            self.layer_stack_for_second_block.to(device=device)
            self.position_enc.to(device=device)

        
        # for the 1st block
        self.embedding_1 = nn.Linear(actual_c_out, self.d_ff)
        self.reduce_dim_z = nn.Linear(self.d_ff, self.c_out)
        # for the 2nd block
        self.embedding_2 = nn.Linear(actual_c_out, self.d_ff)
        self.reduce_dim_beta = nn.Linear(self.d_ff, self.c_out)
        self.reduce_dim_gamma = nn.Linear(self.c_out, self.c_out)
        # for the 3rd block
        self.weight_combine = nn.Linear(self.c_out + self.seq_len, self.c_out)

        self.fusion_layers = self._build_fusion_layers()

    def _build_fusion_layers(self):
        fusion_layers_dict = {
            'WithoutFusion': WithoutFusion,
            'V_DAB': V_DAB,
            'ChannelAttention': ChannelAttention,
            'GroupConv': GroupConv,
            'STAR': STAR,
            'ShuffleConv': ShuffleConv,
            'TSConv2d': TSConv2d,
            'TSDeformConv2d': TSDeformConv2d,
        }
        fusion_layers = fusion_layers_dict[self.configs.last_fusion](self.configs).float()

        if self.configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            fusion_layers.to(device=device)
        return fusion_layers


    def forward(self, x_enc, mask=None):
        dec_out = self.imputation(x_enc, mask)
        return dec_out

    def imputation(self, x_enc, mask):
        # * B is batch size
        # * L is the len of the seq
        # * M is the spatial data nums
        # * S is the source nums
        B, L, M, S = x_enc.shape

        dec_outs = torch.zeros(B, L, M, S, device=x_enc.device)

        for source_idx in range(S):
            # Select right now data
            x_enc_source = x_enc[..., source_idx]
            mask_source = mask[..., source_idx]
            x_enc_source = x_enc_source.masked_fill(mask_source == 0, 0)

            input_X_for_first = torch.cat([x_enc_source, mask_source], dim=2)
            input_X_for_first = self.embedding_1(input_X_for_first)
            enc_output = self.dropout(
                self.position_enc(input_X_for_first)
            )  # namely term e in math algo
            if self.param_sharing_strategy == "between_group":
                for _ in range(self.n_groups):
                    for encoder_layer in self.layer_stack_for_first_block:
                        enc_output, _ = encoder_layer(enc_output)
            else:
                for encoder_layer in self.layer_stack_for_first_block:
                    for _ in range(self.n_group_inner_layers):
                        enc_output, _ = encoder_layer(enc_output)

            X_tilde_1 = self.reduce_dim_z(enc_output)
            X_prime = mask_source * x_enc_source + (1 - mask_source) * X_tilde_1

            # the second DMSA block
            input_X_for_second = (torch.cat([X_prime, mask_source], dim=2))
            input_X_for_second = self.embedding_2(input_X_for_second)
            enc_output = self.position_enc(
                input_X_for_second
            )  # namely term alpha in math algo
            if self.param_sharing_strategy == "between_group":
                for _ in range(self.n_groups):
                    for encoder_layer in self.layer_stack_for_second_block:
                        enc_output, attn_weights = encoder_layer(enc_output)
            else:
                for encoder_layer in self.layer_stack_for_second_block:
                    for _ in range(self.n_group_inner_layers):
                        enc_output, attn_weights = encoder_layer(enc_output)

            X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

            # the attention-weighted combination block
            attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
            if len(attn_weights.shape) == 4:
                # if having more than 1 head, then average attention weights from all heads
                attn_weights = torch.transpose(attn_weights, 1, 3)
                attn_weights = attn_weights.mean(dim=3)
                attn_weights = torch.transpose(attn_weights, 1, 2)

            combining_weights = F.sigmoid(
                self.weight_combine(torch.cat([mask_source, attn_weights], dim=2))
            )  # namely term eta
            # combine X_tilde_1 and X_tilde_2
            X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
            # replace non-missing part with original data
            X_c = mask_source * x_enc_source + (1 - mask_source) * X_tilde_3

            # Store in dec_outs
            dec_outs[..., source_idx] = X_c

        B, L, M, S = dec_outs.shape
        dec_outs = self.fusion_layers(dec_outs)
        return dec_outs