"""
BrainECHO model adapted from:
[BrainECHO: Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation](https://aclanthology.org/2025.findings-acl.142/) (Li et al., Findings 2025)
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import Mapping, Text, Tuple
from torch.cuda.amp import autocast
from einops.einops import rearrange
from einops.layers.torch import Rearrange
from argparse import Namespace
from utils.utils import ssim

from loss.ssim import MS_SSIM_L1_mix_loss
from loss.clip import CLIPLoss
from modules.vae import Encoder, Decoder
from peft import AdaLoraModel
from biot_ablation import (AblatedBIOTEncoder, _PatchFrequencyEmbedding,
                            _PositionalEncoding)

def check_nan_inf(input_data, text):
    nan_check = torch.isnan(input_data)
    if nan_check.any().item():
        print(text, "NaN")

    inf_check = torch.isinf(input_data)
    if inf_check.any().item():
        print(text, "Inf")


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, cls_token=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (5000,840)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (420,) # 840/2
        pe[:, 0::2] = torch.sin(position * div_term)  #
        pe[:, 1::2] = torch.cos(position * div_term)
        if cls_token:
            pe = torch.cat([torch.zeros((1, d_model)), pe], 0)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (5000,1,840)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000, cls_token=False):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        scale = d_model**-0.5
        self.pe = nn.Parameter(scale * torch.randn(max_len, d_model))
        if cls_token:
            class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
            self.pe = torch.cat([class_embedding, self.pe], 0)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


class NeuSpeech(nn.Module):
    def __init__(self, pretrained_layers):
        super(NeuSpeech, self).__init__()
        self.pretrained = pretrained_layers

    def forward(self, rawEEG_batch, target_ids_batch_converted, decoder_input_ids_batch=None):
        b, c, t = rawEEG_batch.shape
        if t < 3000:
            pad = torch.zeros((b, c, 3000 - t)).to(rawEEG_batch.device)
            rawEEG_batch = torch.cat([rawEEG_batch, pad], -1)
            del pad
        elif t > 3000:
            rawEEG_batch = rawEEG_batch[..., :3000]

        out = self.pretrained(
            input_features=rawEEG_batch,
            return_dict=True,
            labels=target_ids_batch_converted[:, 1:],
            decoder_input_ids=decoder_input_ids_batch,
        )
        return out

    @torch.no_grad()
    def generate(
        self,
        rawEEG_batch,
        target_ids_batch_converted,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        b, c, t = rawEEG_batch.shape
        if t < 3000:
            pad = torch.zeros((b, c, 3000 - t)).to(rawEEG_batch.device)
            rawEEG_batch = torch.cat([rawEEG_batch, pad], -1)
            del pad
        elif t > 3000:
            rawEEG_batch = rawEEG_batch[..., :3000]


        output = self.pretrained.generate(
            input_features=rawEEG_batch,
            labels=target_ids_batch_converted[:, 1:],
            return_dict=True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

        return Namespace(predictions=output)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        if downsample is None and in_channel != out_channel:
            self.downsample = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, stride), nn.BatchNorm2d(out_channel))

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------------------------------
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class VectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
    ):
        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = rearrange(z, "b h w c -> (b h w) c")

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, embedding.T)
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((z_quantized - z.detach()) ** 2)

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, "b h w c -> b c h w").contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(
                z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3]
            ),
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum("bd,dn->bn", indices, self.embedding.weight)
        else:
            raise NotImplementedError
        return z_quantized


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w->b (h w) c")
        K, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(
            torch.sum(self.embedding**2, dim=1) + torch.sum(x_flat**2, dim=1, keepdim=True),
            x_flat,
            self.embedding.t(),
            alpha=-2.0,
            beta=1.0,
        )

        indices = torch.argmin(distances.float(), dim=-1)

        encodings = F.one_hot(indices, K).float()


        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:

            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + D * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)

            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)


        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        quantized = rearrange(quantized, "b (h w) c->b c h w", h=h)

        result_dict = dict(quantizer_loss=loss, min_encoding_indices=indices.view(b, h, w))

        return quantized, result_dict


class MelVQVAE(nn.Module):
    def __init__(self, num_res_blocks=0, codebook_size=2048, latent_channels=8, downsample_ratio=4):
        super(MelVQVAE, self).__init__()

        ch_mult = (1,) + (2,) * (int(math.log2(downsample_ratio)) - 1) + (4,)

        self.quantizer = VectorQuantizer(codebook_size, latent_channels, use_l2_norm=True)
        self.codebook_size = codebook_size

        self.mel_encoder = Encoder(ch=32, ch_mult=ch_mult, num_res_blocks=1, double_z=False)
        self.mel_decoder = Decoder(ch=32, ch_mult=ch_mult, num_res_blocks=num_res_blocks, z_channels=latent_channels)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights.

        Args:
            module (torch.nn.Module): module to initialize
        """        
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, mel):
        mel = mel.unsqueeze(1)

        return self.mel_encoder(mel)

    def quantize(self, z):
        z_quantized, result_dict = self.quantizer(z)
        return z_quantized, result_dict

    def decode(self, z):
        x = self.mel_decoder(z)
        x = rearrange(x, "b 1 d t->b d t")

        return x

    def forward(
        self,
        target_mel_feature_batch,
        loss="mse",
    ):
        z = self.encode(target_mel_feature_batch)
        z_quantized, result_dict = self.quantize(z)
        total_loss = 0.5 * result_dict["quantizer_loss"]
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if loss == "mse":
            criterion = nn.MSELoss(reduction="none")
            mel_loss = criterion(recon_mel, target_mel_feature_batch)
            mel_loss = mel_loss.mean()
            total_loss += mel_loss
        elif loss == "ssim":
            sim = ssim(recon_mel.unsqueeze(1), target_mel_feature_batch.unsqueeze(1))
            mel_loss = 1 - sim
            total_loss += mel_loss
        elif loss == "ms_ssim_l1_mix":
            criterion = MS_SSIM_L1_mix_loss(win_size=5)
            encoded_embedding_img = recon_mel.unsqueeze(1).repeat(1, 3, 1, 1)
            target_mel_feature_batch = target_mel_feature_batch.unsqueeze(1).repeat(1, 3, 1, 1)
            mel_loss = criterion(encoded_embedding_img, target_mel_feature_batch).mean()
            total_loss += mel_loss
        else:
            raise Exception("loss not found")
        return Namespace(loss=total_loss, mel_loss=mel_loss, recon_mel=recon_mel, indices=indices)


class BrainECHO(nn.Module):
    def __init__(
        self,
        pretrained_layers,
        in_chan=64,
        d_model=256,
        n_head=8,
        ffn_dim=2048,
        num_layers=4,
        mel_len=1200,
        codebook_size=2048,
        latent_channels=8,
        r=4,
        use_mlp=False,
        mel_interp=False,
        split_mel=False,
    ):
        super(BrainECHO, self).__init__()

        self.task = None
        self.pretrained = pretrained_layers
        self.mel_len = mel_len
        self.mel_interp = mel_interp
        self.split_mel = split_mel

        self.TS_Conv = nn.Sequential(
            nn.Conv2d(1, 64, (1, 5), (1, 2), (0, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, d_model, (in_chan, 1), 1),
            nn.BatchNorm2d(d_model),
            nn.ELU(inplace=True),
        )
        for i in range((int(math.log2(r)) - 2)):
            self.TS_Conv.add_module(
                f"conv_block_{i}",
                nn.Sequential(
                    nn.Conv2d(d_model, d_model, (1, 3), (1, 2), (0, 1)),
                    nn.BatchNorm2d(d_model),
                    nn.ELU(inplace=True),
                ),
            )
        self.TS_Conv.add_module("rearrange", Rearrange("b c h w->b (h w) c"))

        eeg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(eeg_encoder_layer, num_layers=num_layers)
        self.pos_embed_e = LearnablePositionalEncoding(d_model, max_len=mel_len // r * 2)
        self.mel_vqvae = MelVQVAE(codebook_size=codebook_size, downsample_ratio=r)
        for p in self.mel_vqvae.parameters():
            p.requires_grad = False
        self.mel_vqvae.eval()

        if r > 2:
            self.conv_out = nn.Conv2d(1, latent_channels, (1, 3), (1, 2), (0, 1))
        else:
            self.conv_out = nn.Conv2d(1, latent_channels, (1, 3), (1, 1), (0, 1))
        self.r = r

        self.fc_eeg = nn.Linear(d_model, 80 // r)
        if not use_mlp:
            self.mid_fc = ResidualAdd(
                nn.Sequential(nn.Linear(80 // r * mel_len // r, 80 // r * mel_len // r), nn.GELU())
            )
        else:
            self.mid_fc = ResidualAdd(
                nn.Sequential(
                    nn.Linear(80 // r * mel_len // r, ffn_dim),
                    nn.LayerNorm(ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, 80 // r * mel_len // r),
                )
            )
        self.ln = nn.LayerNorm(80 // r * mel_len // r)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights.

        Args:
            module (torch.nn.Module): module to initialize
        """        
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def add_adalora_pretrained_model(self, pretrained, config):
        pretrained.model.encoder = AdaLoraModel(pretrained.model.encoder, config, "default")
        self.pretrained = pretrained

    def encode(self, rawEEG):
        rawEEG = rawEEG.unsqueeze(1)
        eeg_embedding = self.TS_Conv(rawEEG)
        eeg_embedding = self.pos_embed_e(eeg_embedding)
        eeg_embedding = self.transformer(eeg_embedding)
        eeg_embedding = F.gelu(self.fc_eeg(eeg_embedding))
        eeg_embedding = rearrange(eeg_embedding, "b l d->b 1 d l")
        eeg_embedding = F.elu(self.conv_out(eeg_embedding), inplace=True)
        eeg_embedding = rearrange(eeg_embedding, "b c h w->b c (h w)")
        eeg_embedding = self.mid_fc(eeg_embedding)
        eeg_embedding = self.ln(eeg_embedding)
        eeg_embedding = rearrange(eeg_embedding, "b c (h w)->b c h w", h=80 // self.r)

        return eeg_embedding

    def decode(self, z):
        x = self.mel_vqvae.decode(z)

        return x

    def forward(
        self,
        rawEEG_batch,
        target_ids_batch_converted,
        target_mel_feature_batch,
        decoder_input_ids_batch=None,
        loss="mse",
    ):
        """_summary_

        Args:
            rawEEG_batch (Tensor): batch_size * channels * time
            target_ids_batch_converted (Tensor): batch_size * max_length
            target_mel_feature_batch (Tensor): batch_size * 80 * time_frames
            decoder_input_ids_batch (Tensor, optional): batch_size * max_length. Defaults to None.
            loss (str, optional): latent alignment criterion. Defaults to "mse".

        Returns:
            Namespace: loss, reconstructed mel spectrograms or codebook indices
        """        

        z = self.encode(rawEEG_batch)
        z_quantized, result_dict = self.mel_vqvae.quantize(z)
        total_loss = 0.5 * result_dict["quantizer_loss"]
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if self.task == "task1":
            with torch.no_grad():
                mel_z = self.mel_vqvae.encode(target_mel_feature_batch)
            if loss == "mse":
                latent_loss = F.mse_loss(z, mel_z)
            elif loss == "cos_sim":
                z = rearrange(z, "b c h w->(b h w) c")
                mel_z = rearrange(mel_z, "b c h w->(b h w) c")
                target = torch.ones(z.shape[0]).to(z.device)
                latent_loss = F.cosine_embedding_loss(z, mel_z, target)
            elif loss == "clip_loss":
                criterion = CLIPLoss()
                z = rearrange(z, "b c h w->(b h) w c")
                mel_z = rearrange(mel_z, "b c h w->(b h) w c")
                latent_loss = criterion(z, mel_z)
            else:
                raise Exception("loss not found")

            del mel_z
            mel_loss = F.mse_loss(recon_mel, target_mel_feature_batch)
            total_loss = total_loss + latent_loss + mel_loss

            return Namespace(loss=total_loss, mel_loss=mel_loss, recon_mel=recon_mel, indices=indices)

        elif self.task == "task2":
            if self.mel_interp:
                recon_mel = F.interpolate(recon_mel.unsqueeze(1), (80, self.mel_len * 2), mode="bilinear").squeeze(1)
            if self.split_mel:
                recon_mel = rearrange(recon_mel, "(b p) d t->b d (p t)", p=1200 // self.mel_len)
            
            b, d, t = recon_mel.shape
            # pad spectrogram to 3000 time slots
            pad = -1 * torch.ones((b, d, 3000 - t)).to(recon_mel.device)
            recon_mel = torch.cat([recon_mel, pad], -1)

            out = self.pretrained(
                input_features=recon_mel,
                return_dict=True,
                labels=target_ids_batch_converted[:, 1:],
                decoder_input_ids=decoder_input_ids_batch,
            )
            return Namespace(loss=out.loss, logits=out.logits, indices=indices)

    @torch.no_grad()
    def generate(
        self,
        rawEEG_batch,
        target_ids_batch_converted,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        z = self.encode(rawEEG_batch)
        z_quantized, result_dict = self.mel_vqvae.quantize(z)
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if self.mel_interp:
            recon_mel = F.interpolate(recon_mel.unsqueeze(1), (80, self.mel_len * 2), mode="bilinear").squeeze(1)
        if self.split_mel:
            recon_mel = rearrange(recon_mel, "(b p) d t->b d (p t)", p=1200 // self.mel_len)
        b, d, t = recon_mel.shape
        pad = -1 * torch.ones((b, d, 3000 - t)).to(recon_mel.device)
        recon_mel = torch.cat([recon_mel, pad], -1)

        output = self.pretrained.generate(
            input_features=recon_mel,
            labels=target_ids_batch_converted[:, 1:],
            return_dict=True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

        return Namespace(predictions=output, recon_mel=recon_mel, indices=indices)

class BrainECHOBIOT(nn.Module):
    """
    BrainECHO with BIOT STFT encoder replacing TS_Conv + transformer.

    The EEG encoder frontend (TS_Conv, pos_embed_e, transformer) is replaced
    with AblatedBIOTEncoder (use_stft=True), which is mathematically length-
    invariant: zero-padded regions produce zero STFT coefficients and do not
    contribute to the EEG representation.

    BIOT produces (B, C*ts, emb_size) tokens which are adaptively pooled to
    (B, mel_len//r, d_model) to match the downstream fc_eeg input. All layers
    from fc_eeg onwards are identical to the original BrainECHO.
    """

    def __init__(
        self,
        pretrained_layers,
        in_chan=64,
        d_model=256,
        n_head=8,
        ffn_dim=2048,
        num_layers=4,
        mel_len=1200,
        codebook_size=2048,
        latent_channels=8,
        r=4,
        use_mlp=False,
        mel_interp=False,
        split_mel=False,
        # BIOT-specific
        n_fft=200,
        hop_length=100,
        biot_use_channel_tokens=True,
        biot_use_positional_enc=True,
    ):
        super(BrainECHOBIOT, self).__init__()

        self.task = None
        self.pretrained = pretrained_layers
        self.mel_len = mel_len
        self.mel_interp = mel_interp
        self.split_mel = split_mel
        self.r = r

        # ── BIOT EEG encoder (replaces TS_Conv + pos_embed_e + transformer) ──
        self.biot_encoder = AblatedBIOTEncoder(
            n_chans=in_chan,
            emb_size=d_model,
            num_heads=n_head,
            n_layers=num_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            use_stft=True,              # always True — STFT is the fix
            use_channel_tokens=biot_use_channel_tokens,
            use_positional_enc=biot_use_positional_enc,
        )

        # Pool to 2× the final spatial dim so conv_out (stride 2) produces mel_len//r tokens
        # matching TS_Conv's downsampling behaviour in original BrainECHO
        self.temporal_pool = nn.AdaptiveAvgPool1d(mel_len // r * 2)

        # ── Downstream layers identical to BrainECHO ──
        self.mel_vqvae = MelVQVAE(codebook_size=codebook_size, downsample_ratio=r)
        for p in self.mel_vqvae.parameters():
            p.requires_grad = False
        self.mel_vqvae.eval()

        if r > 2:
            self.conv_out = nn.Conv2d(1, latent_channels, (1, 3), (1, 2), (0, 1))
        else:
            self.conv_out = nn.Conv2d(1, latent_channels, (1, 3), (1, 1), (0, 1))

        self.fc_eeg = nn.Linear(d_model, 80 // r)
        if not use_mlp:
            self.mid_fc = ResidualAdd(
                nn.Sequential(
                    nn.Linear(80 // r * mel_len // r, 80 // r * mel_len // r),
                    nn.GELU()
                )
            )
        else:
            self.mid_fc = ResidualAdd(
                nn.Sequential(
                    nn.Linear(80 // r * mel_len // r, ffn_dim),
                    nn.LayerNorm(ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, 80 // r * mel_len // r),
                )
            )
        self.ln = nn.LayerNorm(80 // r * mel_len // r)

        self.apply(self._init_weights)
        # Re-initialise BIOT encoder with its own init (overrides apply above)
        self.biot_encoder.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data,
                                                        mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data,
                                                        mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def add_adalora_pretrained_model(self, pretrained, config):
        pretrained.model.encoder = AdaLoraModel(pretrained.model.encoder,
                                                 config, "default")
        self.pretrained = pretrained

    def encode(self, rawEEG):
        # rawEEG: (B, C, T)
        # Step 1: BIOT sequence encoding → (B, C*ts, emb_size)
        eeg_embedding = self.biot_encoder.encode_sequence(rawEEG)

        # Step 2: Adaptive pool over token dimension → (B, emb_size, mel_len//r)
        eeg_embedding = eeg_embedding.permute(0, 2, 1)        # (B, emb_size, C*ts)
        eeg_embedding = self.temporal_pool(eeg_embedding)      # (B, emb_size, mel_len//r)
        eeg_embedding = eeg_embedding.permute(0, 2, 1)        # (B, mel_len//r, emb_size)

        # Step 3: fc_eeg + downstream — identical to BrainECHO
        eeg_embedding = F.gelu(self.fc_eeg(eeg_embedding))    # (B, mel_len//r, 80//r)
        eeg_embedding = rearrange(eeg_embedding, "b l d->b 1 d l")
        eeg_embedding = F.elu(self.conv_out(eeg_embedding), inplace=True)
        eeg_embedding = rearrange(eeg_embedding, "b c h w->b c (h w)")
        eeg_embedding = self.mid_fc(eeg_embedding)
        eeg_embedding = self.ln(eeg_embedding)
        eeg_embedding = rearrange(eeg_embedding, "b c (h w)->b c h w",
                                   h=80 // self.r)
        return eeg_embedding

    def decode(self, z):
        return self.mel_vqvae.decode(z)

    def forward(
            self,
            rawEEG_batch,
            target_ids_batch_converted,
            target_mel_feature_batch=None,
            decoder_input_ids_batch=None,
            loss="mse",
    ):
        """Identical interface to BrainECHO.forward()"""
        z = self.encode(rawEEG_batch)
        z_quantized, result_dict = self.mel_vqvae.quantize(z)
        total_loss = 0.5 * result_dict["quantizer_loss"]
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if self.task == "task1":
            with torch.no_grad():
                mel_z = self.mel_vqvae.encode(target_mel_feature_batch)
            if loss == "mse":
                latent_loss = F.mse_loss(z, mel_z)
            elif loss == "cos_sim":
                z_flat = rearrange(z, "b c h w->(b h w) c")
                mel_z_flat = rearrange(mel_z, "b c h w->(b h w) c")
                target = torch.ones(z_flat.shape[0]).to(z_flat.device)
                latent_loss = F.cosine_embedding_loss(z_flat, mel_z_flat, target)
            elif loss == "clip_loss":
                criterion = CLIPLoss()
                z_flat = rearrange(z, "b c h w->(b h) w c")
                mel_z_flat = rearrange(mel_z, "b c h w->(b h) w c")
                latent_loss = criterion(z_flat, mel_z_flat)
            else:
                raise Exception("loss not found")
            del mel_z
            mel_loss = F.mse_loss(recon_mel, target_mel_feature_batch)
            total_loss = total_loss + latent_loss + mel_loss
            return Namespace(loss=total_loss, mel_loss=mel_loss,
                             recon_mel=recon_mel, indices=indices)

        elif self.task == "task2":
            b, d, t = recon_mel.shape
            pad = -1 * torch.ones((b, d, 3000 - t)).to(recon_mel.device)
            recon_mel = torch.cat([recon_mel, pad], -1)
            out = self.pretrained(
                input_features=recon_mel,
                return_dict=True,
                labels=target_ids_batch_converted[:, 1:],
                decoder_input_ids=decoder_input_ids_batch,
            )
            return Namespace(loss=out.loss, logits=out.logits, indices=indices)

    @torch.no_grad()
    def generate(
            self,
            rawEEG_batch,
            target_ids_batch_converted,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=None,
            assistant_model=None,
            streamer=None,
            negative_prompt_ids=None,
            negative_prompt_attention_mask=None,
            **kwargs,
    ):
        z = self.encode(rawEEG_batch)
        z_quantized, result_dict = self.mel_vqvae.quantize(z)
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if self.split_mel:
            recon_mel = rearrange(recon_mel, "(b p) d t->b d (p t)", p=1200 // self.mel_len)
        b, d, t = recon_mel.shape
        pad = -1 * torch.ones((b, d, 3000 - t)).to(recon_mel.device)
        recon_mel = torch.cat([recon_mel, pad], -1)

        output = self.pretrained.generate(
            input_features=recon_mel,
            labels=target_ids_batch_converted[:, 1:],
            return_dict=True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )
        return Namespace(predictions=output, recon_mel=recon_mel, indices=indices)


class BrainECHOSTFT(nn.Module):
    """
    BrainECHO with STFT tokenizer replacing TS_Conv, but keeping
    BrainECHO's own standard TransformerEncoder.

    Comparison:
      BrainECHO      : TS_Conv (time domain)  + BrainECHO TransformerEncoder
      BrainECHO-STFT : STFT tokenizer         + BrainECHO TransformerEncoder
      BrainECHO-BIOT : STFT tokenizer         + BIOT LinearAttentionTransformer

    This isolates the contribution of STFT tokenization independently from
    the choice of transformer architecture.
    """

    def __init__(
        self,
        pretrained_layers,
        in_chan=64,
        d_model=256,
        n_head=8,
        ffn_dim=2048,
        num_layers=4,
        mel_len=1200,
        codebook_size=2048,
        latent_channels=8,
        r=4,
        use_mlp=False,
        mel_interp=False,
        split_mel=False,
        # STFT parameters
        n_fft=200,
        hop_length=100,
        use_channel_tokens=True,
        use_positional_enc=True,
    ):
        super(BrainECHOSTFT, self).__init__()

        self.task = None
        self.pretrained = pretrained_layers
        self.mel_len = mel_len
        self.mel_interp = mel_interp
        self.split_mel = split_mel
        self.r = r
        self.n_fft = int(n_fft)
        self.hop_length = hop_length
        self.in_chan = in_chan
        self.use_channel_tokens = use_channel_tokens
        self.use_positional_enc = use_positional_enc

        # ── STFT tokenizer (from BIOT) ──────────────────────────────────────
        n_freq = int(self.n_fft // 2 + 1)  # 101 for n_fft=200
        self.patch_embedding = _PatchFrequencyEmbedding(
            emb_size=d_model, n_freq=n_freq
        )

        if use_channel_tokens:
            self.channel_tokens = nn.Embedding(in_chan, d_model)
            self.register_buffer(
                "index", torch.arange(in_chan, dtype=torch.long)
            )

        if use_positional_enc:
            self.positional_encoding = _PositionalEncoding(
                emb_size=d_model, max_len=1024
            )

        # ── BrainECHO's own TransformerEncoder (unchanged) ──────────────────
        eeg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ffn_dim,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            eeg_encoder_layer, num_layers=num_layers
        )

        # ── Temporal pooling: C*ts tokens → mel_len//r*2 tokens ─────────────
        self.temporal_pool = nn.AdaptiveAvgPool1d(mel_len // r * 2)

        # ── Downstream layers identical to BrainECHO ────────────────────────
        self.mel_vqvae = MelVQVAE(
            codebook_size=codebook_size, downsample_ratio=r
        )
        for p in self.mel_vqvae.parameters():
            p.requires_grad = False
        self.mel_vqvae.eval()

        if r > 2:
            self.conv_out = nn.Conv2d(1, latent_channels, (1, 3), (1, 2), (0, 1))
        else:
            self.conv_out = nn.Conv2d(1, latent_channels, (1, 3), (1, 1), (0, 1))

        self.fc_eeg = nn.Linear(d_model, 80 // r)
        if not use_mlp:
            self.mid_fc = ResidualAdd(
                nn.Sequential(
                    nn.Linear(
                        80 // r * mel_len // r,
                        80 // r * mel_len // r
                    ),
                    nn.GELU()
                )
            )
        else:
            self.mid_fc = ResidualAdd(
                nn.Sequential(
                    nn.Linear(80 // r * mel_len // r, ffn_dim),
                    nn.LayerNorm(ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, 80 // r * mel_len // r),
                )
            )
        self.ln = nn.LayerNorm(80 // r * mel_len // r)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _stft(self, x):
        """
        STFT per channel.
        x : (B, 1, T)
        returns : (B, n_freq, ts)
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1)          # (B, T)
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            center=False,
        )                                   # (B, n_freq, ts)
        return spec.abs()

    def _tokenise(self, x):
        """
        STFT tokenization per channel.
        x : (B, C, T)
        returns : (B, C*ts, d_model)
        """
        emb_seq = []
        for i in range(x.shape[1]):
            chan = x[:, i:i+1, :]          # (B, 1, T)
            spec = self._stft(chan)         # (B, n_freq, ts)
            tok = self.patch_embedding(spec) # (B, ts, d_model)
            B, ts, E = tok.shape
            if self.use_channel_tokens:
                chan_tok = (
                    self.channel_tokens(self.index[i])
                    .unsqueeze(0).unsqueeze(0)
                    .repeat(B, ts, 1)
                )
                tok = tok + chan_tok
            if self.use_positional_enc:
                tok = self.positional_encoding(tok)
            emb_seq.append(tok)
        return torch.cat(emb_seq, dim=1)   # (B, C*ts, d_model)

    def add_adalora_pretrained_model(self, pretrained, config):
        pretrained.model.encoder = AdaLoraModel(
            pretrained.model.encoder, config, "default"
        )
        self.pretrained = pretrained

    def encode(self, rawEEG):
        # rawEEG: (B, C, T)

        # Step 1: STFT tokenisation → (B, C*ts, d_model)
        eeg_embedding = self._tokenise(rawEEG)

        # Step 2: BrainECHO transformer
        eeg_embedding = self.transformer(eeg_embedding)  # (B, C*ts, d_model)

        # Step 3: Pool to mel_len//r*2 tokens
        eeg_embedding = eeg_embedding.permute(0, 2, 1)           # (B, d_model, C*ts)
        eeg_embedding = self.temporal_pool(eeg_embedding)         # (B, d_model, mel_len//r*2)
        eeg_embedding = eeg_embedding.permute(0, 2, 1)           # (B, mel_len//r*2, d_model)

        # Step 4: Downstream (identical to BrainECHO from fc_eeg onwards)
        eeg_embedding = F.gelu(self.fc_eeg(eeg_embedding))       # (B, mel_len//r*2, 80//r)
        eeg_embedding = rearrange(eeg_embedding, "b l d->b 1 d l")
        eeg_embedding = F.elu(self.conv_out(eeg_embedding), inplace=True)
        eeg_embedding = rearrange(eeg_embedding, "b c h w->b c (h w)")
        eeg_embedding = self.mid_fc(eeg_embedding)
        eeg_embedding = self.ln(eeg_embedding)
        eeg_embedding = rearrange(
            eeg_embedding, "b c (h w)->b c h w", h=80 // self.r
        )
        return eeg_embedding

    def decode(self, z):
        return self.mel_vqvae.decode(z)

    def forward(
        self,
        rawEEG_batch,
        target_ids_batch_converted,
        target_mel_feature_batch=None,
        decoder_input_ids_batch=None,
        loss="mse",
    ):
        z = self.encode(rawEEG_batch)
        z_quantized, result_dict = self.mel_vqvae.quantize(z)
        total_loss = 0.5 * result_dict["quantizer_loss"]
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if self.task == "task1":
            with torch.no_grad():
                mel_z = self.mel_vqvae.encode(target_mel_feature_batch)
            if loss == "mse":
                latent_loss = F.mse_loss(z, mel_z)
            elif loss == "cos_sim":
                z_flat = rearrange(z, "b c h w->(b h w) c")
                mel_z_flat = rearrange(mel_z, "b c h w->(b h w) c")
                target = torch.ones(z_flat.shape[0]).to(z_flat.device)
                latent_loss = F.cosine_embedding_loss(z_flat, mel_z_flat, target)
            elif loss == "clip_loss":
                criterion = CLIPLoss()
                z_flat = rearrange(z, "b c h w->(b h) w c")
                mel_z_flat = rearrange(mel_z, "b c h w->(b h) w c")
                latent_loss = criterion(z_flat, mel_z_flat)
            else:
                raise Exception("loss not found")
            del mel_z
            mel_loss = F.mse_loss(recon_mel, target_mel_feature_batch)
            total_loss = total_loss + latent_loss + mel_loss
            return Namespace(
                loss=total_loss, mel_loss=mel_loss,
                recon_mel=recon_mel, indices=indices
            )

        elif self.task == "task2":
            b, d, t = recon_mel.shape
            pad = -1 * torch.ones((b, d, 3000 - t)).to(recon_mel.device)
            recon_mel = torch.cat([recon_mel, pad], -1)
            out = self.pretrained(
                input_features=recon_mel,
                return_dict=True,
                labels=target_ids_batch_converted[:, 1:],
                decoder_input_ids=decoder_input_ids_batch,
            )
            return Namespace(loss=out.loss, logits=out.logits, indices=indices)

    @torch.no_grad()
    def generate(
        self,
        rawEEG_batch,
        target_ids_batch_converted,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        z = self.encode(rawEEG_batch)
        z_quantized, result_dict = self.mel_vqvae.quantize(z)
        indices = result_dict["min_encoding_indices"]
        recon_mel = self.decode(z_quantized)

        if self.split_mel:
            recon_mel = rearrange(
                recon_mel, "(b p) d t->b d (p t)", p=1200 // self.mel_len
            )
        b, d, t = recon_mel.shape
        pad = -1 * torch.ones((b, d, 3000 - t)).to(recon_mel.device)
        recon_mel = torch.cat([recon_mel, pad], -1)

        output = self.pretrained.generate(
            input_features=recon_mel,
            labels=target_ids_batch_converted[:, 1:],
            return_dict=True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )
        return Namespace(predictions=output, recon_mel=recon_mel, indices=indices)