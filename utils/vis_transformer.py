import torch
import torch.nn as nn

class VisTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=False, memory_is_causal=False):
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross Attention
        tgt2, attn_weights = self.multihead_attn(
            tgt, memory, memory, 
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True, average_attn_weights=True # average over heads
        )
        
        # Shape: [B, Tgt_Len(Query), Src_Len(Key)]
        self.last_attn_weights = attn_weights.detach().cpu()

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed Forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class VisTransformer(nn.Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        decoder_layer = VisTransformerDecoderLayer(
            d_model=kwargs['d_model'], 
            nhead=kwargs['nhead'],
            dim_feedforward=kwargs['dim_feedforward'],
            dropout=kwargs['dropout'],
            batch_first=True 
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, kwargs['num_decoder_layers'])

    def get_last_cross_attention(self):
        # [B, Chunk_Size, Total_Tokens]
        return self.decoder.layers[-1].last_attn_weights