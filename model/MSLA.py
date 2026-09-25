import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import TransformerBlock
from typing import Any, Dict, Optional, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_MODALITIES = ['modal0', 'modal1', 'modal2']
    
class AddPositionEmbs(nn.Module):
    def __init__(self, seq_len: int, hidden_size: int):
        super(AddPositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_size))

    def forward(self, x):
        pos_embedding = self.pos_embedding.to(x.device)
        return x + pos_embedding

def add_positional_embed(x):
    assert x.ndim == 3
    seq_len, emb_dim = x.shape[1], x.shape[2]
    return AddPositionEmbs(seq_len, emb_dim)(x)

class EncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        mlp_dim: int = 3072,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.hidden_size = hidden_size
        self.transformer_layers = TransformerBlock(
            hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn
        )

    def forward(self, x):
        self.transformer_layers = self.transformer_layers.to(x.device)
        x = self.transformer_layers(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        mlp_dim: int = 3072,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        modality_fusion: Tuple[str] = ('modal0', 'modal1', 'modal2',),
        fusion_layer: int = 2,
        use_bottleneck: bool = True,
        share_encoder: bool = False,
        dtype: Any = torch.float32,
    ) -> None:
        super(Encoder, self).__init__()
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.modality_fusion = modality_fusion
        self.fusion_layer = fusion_layer
        self.use_bottleneck = use_bottleneck
        self.share_encoder = share_encoder
        self.dtype = dtype

    def get_encoder_block(self, encoder_block):
        return encoder_block(
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )

    def get_context(self, target_modality, modality_fusion, x):
        context = []
        for modality in _MODALITIES:
            if modality != target_modality and modality in modality_fusion:
                context.append(x[modality])
        return context

    def combine_context(self, x, other_modalities):
        num_tokens = x.shape[1]
        other_modalities.append(x)
        x_combined = torch.cat(other_modalities, dim=1)
        return x_combined, num_tokens

    def forward(self, x: Dict[str, torch.Tensor], bottleneck: Optional[torch.Tensor] = None):
        for modality in self.modality_fusion:
            x[modality] = add_positional_embed(x[modality])

        x_combined = None

        for lyr in range(self.num_layers):
            encoders = {}
            encoders['modal0'] = self.get_encoder_block(EncoderBlock)

            for modality in self.modality_fusion:
                if modality != 'modal0':
                    if self.share_encoder:
                        encoders[modality] = encoders['modal0']
                    else:
                        encoders[modality] = self.get_encoder_block(EncoderBlock)

            if (lyr < self.fusion_layer or len(self.modality_fusion) == 1):
                for modality in self.modality_fusion:
                    x[modality] = encoders[modality](x[modality])
            else:
                if self.use_bottleneck:
                    bottle = []
                    for modality in self.modality_fusion:
                        t_mod = x[modality].shape[1]
                        in_mod = torch.cat([x[modality], bottleneck], dim=1)
                        out_mod = encoders[modality](in_mod)
                        x[modality] = out_mod[:, :t_mod]
                        bottle.append(out_mod[:, t_mod:])
                    bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)
                else:
                    if not self.share_encoder and len(self.modality_fusion) > 1:
                        x_new = {}
                        for modality in self.modality_fusion:
                            other_modalities = self.get_context(modality, self.modality_fusion, x)
                            combined_mods, t = self.combine_context(x[modality], other_modalities)
                            combined_mods = encoders[modality](combined_mods)
                            x_new[modality] = combined_mods[:, -t:]
                        x = x_new

                    elif self.share_encoder and len(self.modality_fusion) > 1:
                        if x_combined is None:
                            x_combined = []
                            for modality in self.modality_fusion:
                                x_combined.append(x[modality])
                            x_combined = torch.cat(x_combined, dim=1)
                        x_combined = encoders['modal0'](x_combined)

        if x_combined is not None:
            x_out = x_combined
        else:
            x_out = []
            for modality in self.modality_fusion:
                x_out.append(x[modality])
            x_out = torch.cat(x_out, dim=1)
        encoded = nn.LayerNorm(x_out.size()[1:]).to(device)(x_out)
        return encoded

class MSLA(nn.Module):
    def __init__(
        self,
        mlp_dim: int = 3072,
        num_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = 5,
        hidden_size: int = 512,
        representation_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        classifier: str = 'token',
        modality_fusion: Tuple[str] = ('modal0','modal1', 'modal2',),
        fusion_layer: int = 2,
        return_prelogits: bool = True,
        return_preclassifier: bool = False,
        use_bottleneck: bool = True, 
        n_bottlenecks: int = 4,
        share_encoder: bool = False,
        dtype: Any = torch.float32,

    ) -> None:
        super(MSLA, self).__init__()
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.representation_size = representation_size
        self.dropout_rate = dropout_rate
        self.classifier = classifier
        self.modality_fusion = modality_fusion
        self.fusion_layer = fusion_layer
        self.return_prelogits = return_prelogits
        self.return_preclassifier = return_preclassifier
        self.use_bottleneck = use_bottleneck
        self.n_bottlenecks = n_bottlenecks
        self.share_encoder = share_encoder
        self.dtype = dtype

        self.encoder = Encoder(
            modality_fusion=self.modality_fusion,
            fusion_layer=self.fusion_layer,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            use_bottleneck=self.use_bottleneck,
            share_encoder=self.share_encoder,
        )

        self.cls_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            for modality in self.modality_fusion
        })

        if self.use_bottleneck:
            self.bottleneck = nn.Parameter(
                torch.randn(1, self.n_bottlenecks + (1 if self.classifier == 'token' else 0), self.hidden_size)
            )
        self.output_projection = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, main_f, co_f1, co_f2):
        x = {}
        x['modal0'] = add_positional_embed(main_f)
        x['modal1'] = add_positional_embed(co_f1)
        x['modal2'] = add_positional_embed(co_f2)

        temporal_dims = {}
        for modality in self.modality_fusion:
            if self.classifier == 'token':
                n, temporal_dims[modality], c = x[modality].shape
                cls_token = self.cls_tokens[modality].expand(n, -1, -1)
                x[modality] = torch.cat([cls_token, x[modality]], dim=1)

        bottleneck = None
        if self.use_bottleneck:
            bottleneck = self.bottleneck.expand(x['modal0'].size(0), -1, -1)
        x = self.encoder(x, bottleneck)

        if self.return_preclassifier:
            return x

        if self.classifier == 'token':
            x_out = []
            counter = 0
            for modality in self.modality_fusion:
                x_out.append(x[:, counter])
                counter += temporal_dims[modality] + 1
            x_out = torch.cat(x_out, dim=1)
    
        else:
            if self.classifier in ('gap', 'gmp', 'gsp'):
                fn = {'gap': torch.mean, 'gmp': torch.max, 'gsp': torch.sum}[self.classifier]
                if self.classifier == 'gmp': 
                    x_out = fn(x, dim=1).values
                else:
                    x_out = fn(x, dim=1)

        if self.representation_size is not None:
            pre_logits_fc = nn.Linear(self.hidden_size, self.representation_size).to(device)
            x_out = pre_logits_fc(x_out)
            x_out = torch.tanh(x_out)
        else:
            if not isinstance(x_out, dict):
                x_out = x_out

        if self.return_prelogits:
            return x_out

        x_out = self.output_projection(x_out)
        return x_out


