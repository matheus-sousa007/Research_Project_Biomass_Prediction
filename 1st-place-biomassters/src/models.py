import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class AttentionPooling(nn.Module):
    def __init__(self, embedding_size, hid=None):
        super().__init__()

        hid = embedding_size if hid is None else hid

        self.attn = nn.Sequential(
            nn.Linear(embedding_size, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x, bs, mask=None):
        # x: [B*T, D, H, W]
        # mask: [B, T]
        _, d, h, w = x.size()
        x = x.view(bs, -1, d, h, w)
        x = x.permute(0, 1, 3, 4, 2)  # [bs, t, h, w, d]

        # x:    [B, T, *, D]
        attn_logits = self.attn(x)  # [B, T, *, 1]
        if mask is not None:
            attn_logits[mask] = -torch.inf

        attn_weights = attn_logits.softmax(dim=1)  # [B, T, *, 1]

        x = attn_weights * x  # [B, T, *, D]
        x = x.sum(dim=1)   # [B, *, D]

        x = x.permute(0, 3, 1, 2)     # [bs, d, h, w]

        return x, attn_weights


class TimmEncoder(nn.Module):
    def __init__(self, cfg, output_stride=32):
        super().__init__()

        depth = len(cfg.out_indices)

        self.is_vit = cfg.backbone.startswith("vit") or cfg.backbone.startswith("deit") 

        if self.is_vit:
            self.model = timm.create_model(
                cfg.backbone,
                in_chans=cfg.in_channels,
                pretrained=True,
                num_classes=0,
                img_size=cfg.img_size[0]
            )

            self._in_channels = cfg.in_channels

            embed_dim = self.model.embed_dim

            self._out_channels = [ cfg.in_channels ] + [embed_dim] * depth

            num_blocks = len(self.model.blocks)
            self.vit_indices = [ int(num_blocks / depth * (i + 1)) - 1 for i in range(depth) ]
        else:

            self.model = timm.create_model(
                cfg.backbone,
                in_chans=cfg.in_channels,
                pretrained=True,
                num_classes=0,
                features_only=True,
                output_stride=output_stride if output_stride != 32 else None,
                out_indices=cfg.out_indices if not cfg.backbone.startswith("vit") else None,
            )
            self._in_channels = cfg.in_channels
            
            self._out_channels = [
                cfg.in_channels,
            ] + self.model.feature_info.channels()

            self._depth = depth
            self._output_stride = output_stride  # 32

    def _forward_vit_legacy(self, x):
        """Método de fallback para versões antigas do TIMM"""
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        
        dist_token = getattr(self.model, 'dist_token', None)

        if dist_token is None: 
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            
        x = self.model.pos_drop(x + self.model.pos_embed)

        features = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.vit_indices:
                # Remove tokens extras (CLS, Dist)
                num_extra = 2 if dist_token is not None else 1
                x_spatial = x[:, num_extra:] 
                
                # Reshape (Batch, N_patches, Dim) -> (Batch, Dim, H, W)
                B, N, C = x_spatial.shape
                size = int(N ** 0.5) # Assume imagem quadrada
                feat = x_spatial.permute(0, 2, 1).reshape(B, C, size, size)
                features.append(feat)
        return features

    def forward(self, x):
        if self.is_vit:
            # Tenta usar o método novo. Se não existir (AttributeError), usa o legado.
            if hasattr(self.model, 'get_intermediate_layers'):
                features_list = self.model.get_intermediate_layers(x, n=self.vit_indices, reshape=True)
            else:
                features_list = self._forward_vit_legacy(x)
            
            # --- Bloco de Interpolação (Pirâmide Artificial) ---
            out_features = [x]
            # Create a feature pyramid from the extracted ViT features
            # Adjust the number of pyramid levels based on available features
            if len(features_list) >= 5:
                # for 5 features, create a 5-level pyramid
                f1 = F.interpolate(features_list[0], scale_factor=4, mode='bilinear', align_corners=False)
                f2 = F.interpolate(features_list[1], scale_factor=2, mode='bilinear', align_corners=False)
                f3 = features_list[2]
                f4 = F.max_pool2d(features_list[3], kernel_size=2, stride=2)
                f5 = F.max_pool2d(features_list[4], kernel_size=4, stride=4)
                out_features.extend([f1, f2, f3, f4, f5])
            elif len(features_list) >= 4:
                # For 4 level features, create a 4-level pyramid
                f1 = F.interpolate(features_list[0], scale_factor=4, mode='bilinear', align_corners=False)
                f2 = F.interpolate(features_list[1], scale_factor=2, mode='bilinear', align_corners=False)
                f3 = features_list[2]
                f4 = F.max_pool2d(features_list[3], kernel_size=2, stride=2)
                out_features.extend([f1, f2, f3, f4])
            else:
                # Fallback: just use features as-is
                out_features.extend(features_list)

            return out_features

        else:
            features = self.model(x)
            return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


class UnetVFLOW(nn.Module):
    def __init__(self, args, decoder_use_batchnorm: bool = True):
        super().__init__()

        encoder_name = args.backbone

        self.is_vit = args.backbone.startswith("vit") or args.backbone.startswith("deit") 

        self.encoder = TimmEncoder(args)

        encoder_depth = len(self.encoder.out_channels) - 1

        self.attn = nn.ModuleList(
            [
                AttentionPooling(i)
                for i in self.encoder.out_channels[1:]
            ]
        )

        decoder_channels = args.dec_channels[:encoder_depth]
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=args.dec_attn_type,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], args.n_classes, kernel_size=3, padding=1)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x, mask):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        bs, _, d, h, w = x.size()
        x = x.view(-1, d, h, w)
        x = x.to(memory_format=torch.channels_last)
        features = self.encoder(x)

        if self.is_vit:
            features = [features[0]] + [
                attn(f, bs, mask)[0]
                for f, attn in zip(features[1:], self.attn)  
            ]
        else:
            features = [None] + [
                attn(f, bs, mask)[0]
                for f, attn in zip(features, self.attn)
            ]

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks
