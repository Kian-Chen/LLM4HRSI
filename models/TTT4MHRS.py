import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.iEncoderLayer import EncoderLayer
from layers.embed import DataEmbedding_inverted
from layers.residual_block import PositionalEncoding
from layers.last_fusion import (WithoutFusion, V_DAB, ChannelAttention,
                                GroupConv, STARm, STARc, ShuffleConv,
                                TSConv2d, TSDeformConv2d)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.seq_len = configs.seq_len
        self.len_dff = configs.len_dff
        self.c_out = configs.c_out
        self.d_ff = configs.d_ff
        self.nvars = len(configs.source_names)
        self.n_heads = configs.n_heads
        actual_c_out = self.c_out * 2
        self.d_lower = configs.d_lower
        self.dropout = configs.dropout

        self.is_invert = configs.is_invert

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

        self.mask_fusion1 = nn.Linear(actual_c_out, self.c_out)
        self.mask_fusion2 = nn.Linear(actual_c_out, self.c_out)

        if self.is_invert:
            # for the encoder block
            self.embedding_1 = nn.Linear(self.seq_len, self.len_dff)
            self.embedding_2 = nn.Linear(self.seq_len, self.len_dff)

            self.reduce_dim_1 = nn.Linear(self.len_dff, self.seq_len)
            self.reduce_dim_2 = nn.Linear(self.len_dff, self.seq_len)

            self.position_enc1 = DataEmbedding_inverted(configs.seq_len, configs.len_dff)
            self.position_enc2 = DataEmbedding_inverted(configs.seq_len, configs.len_dff)
        else:
            # for the encoder block
            self.embedding_1 = nn.Linear(self.c_out, self.d_ff)
            self.embedding_2 = nn.Linear(self.c_out, self.d_ff)
            self.reduce_dim_1 = nn.Linear(self.d_ff, self.c_out)
            self.reduce_dim_2 = nn.Linear(self.d_ff, self.c_out)

            self.position_enc = PositionalEncoding(self.d_ff, n_position=self.seq_len)

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.layer_stack_for_first_block.to(device=device)
            self.layer_stack_for_second_block.to(device=device)

        self.expert_nums = len(self.configs.expert_ids)

        # self.fusion_layers = self._build_fusion_layers()
        self.fusion_layers = nn.ModuleList([
            self._build_fusion_layers(name) for name in self.configs.expert_names
            ])

        self.gate = nn.Linear(self.seq_len, self.expert_nums)
        self.softmax = nn.Softmax(dim=-1)

    def _build_fusion_layers(self, layer_name=None):
        fusion_layers_dict = {
            'WithoutFusion': WithoutFusion,
            'V_DAB': V_DAB,
            'ChannelAttention': ChannelAttention,
            'GroupConv': GroupConv,
            'STARm': STARm,
            'STARc': STARc,
            'ShuffleConv': ShuffleConv,
            'TSConv2d': TSConv2d,
            'TSDeformConv2d': TSDeformConv2d,
        }
        if layer_name is None:
            fusion_layer = fusion_layers_dict[self.configs.last_fusion](self.configs).float()
        else:
            fusion_layer = fusion_layers_dict[layer_name](self.configs).float()

        if self.configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            fusion_layer.to(device=device)
        return fusion_layer

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

            # the first encoder block
            input_X_for_first = torch.cat([x_enc_source, mask_source], dim=2)
            input_X_for_first = self.mask_fusion1(input_X_for_first)

            if self.is_invert:
                enc_output = self.dropout(
                    self.position_enc1(input_X_for_first)
                )
            else:
                input_X_for_first = self.embedding_1(input_X_for_first)
                enc_output = self.dropout(
                    self.position_enc(input_X_for_first)
                )

            if self.param_sharing_strategy == "between_group":
                for _ in range(self.n_groups):
                    for encoder_layer in self.layer_stack_for_first_block:
                        enc_output, _ = encoder_layer(enc_output)
            else:
                for encoder_layer in self.layer_stack_for_first_block:
                    for _ in range(self.n_group_inner_layers):
                        enc_output, _ = encoder_layer(enc_output)

            if self.is_invert:
                X_tilde_1 = self.reduce_dim_1(enc_output).permute(0, 2, 1)
            else:
                X_tilde_1 = self.reduce_dim_1(enc_output)
            X_prime = mask_source * x_enc_source + (1 - mask_source) * X_tilde_1

            # the second decoder block
            input_X_for_second = torch.cat([X_prime, mask_source], dim=2)
            input_X_for_second = self.mask_fusion2(input_X_for_second)

            if self.is_invert:
                enc_output = self.dropout(
                    self.position_enc2(input_X_for_second)
                )
            else:
                input_X_for_second = self.embedding_2(input_X_for_second)
                enc_output = self.dropout(
                    self.position_enc(input_X_for_second)
                )

            if self.param_sharing_strategy == "between_group":
                for _ in range(self.n_groups):
                    for encoder_layer in self.layer_stack_for_second_block:
                        enc_output, _ = encoder_layer(enc_output)
            else:
                for encoder_layer in self.layer_stack_for_second_block:
                    for _ in range(self.n_group_inner_layers):
                        enc_output, _ = encoder_layer(enc_output)

            if self.is_invert:
                X_tilde_2 = self.reduce_dim_2(enc_output).permute(0, 2, 1)
            else:
                X_tilde_2 = self.reduce_dim_2(enc_output)
            X_out = mask_source * x_enc_source + (1 - mask_source) * X_tilde_2

            dec_outs[..., source_idx] = X_out

        B, L, M, S = dec_outs.shape

        tmp = dec_outs.permute(0, 2, 3, 1)
        tmp = tmp.reshape(B*M*S, L).contiguous()  # (B*M*S, L)

        score = F.softmax(self.gate(tmp), dim=-1)  # (B*M*S, E)

        # Expert outputs
        expert_outputs = torch.stack([self.fusion_layers[i](dec_outs).permute(0, 2, 3, 1).reshape(B*M*S, L)
                                      for i in range(self.expert_nums)], dim=-1)  # (BxM*S, L, E)

        prediction = torch.einsum("BLE,BE->BL", expert_outputs, score)
        prediction = prediction.reshape(B, M, S, -1).permute(0, 3, 1, 2)
        return prediction