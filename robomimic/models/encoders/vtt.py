import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from torchvision import transforms
from robomimic.models.base_nets import Module
from robomimic.models.transformers import SelfAttentionBlock, CausalSelfAttention, PositionalEncoding

import warnings
import math

# This architecture is DEPENDENT on the context length (i.e. num RGB views, input modalities, etc.)
class VisuotactileTransformer(Module):
    def __init__(
        self,
        img_sizes=(84,),
        img_patch_size=14,
        tactile_dim=12,
        tactile_patches=2,          # Only used for original VTT (no history)
        tactile_history=32,         # Set to 1 for original VTT
        in_channels=3,
        embed_dim=384,
        output_dim=288,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.output_dim = output_dim
        self.img_patch_size = img_patch_size
        self.tactile_patches = tactile_patches
        self.tactile_history = tactile_history
        self.num_rgb = len(img_sizes)

        # Patch embedding networks for separate modalities
        self.rgb_patch_embed = RGBPatchEmbed(
            img_size=img_sizes[0],
            img_patch_size=img_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        if tactile_history == 1:
            self.tactile_patch_embed = TactilePatchEmbedNoHistory(
                tactile_dim=tactile_dim,
                num_tactile_patches=tactile_patches,
                embed_dim=embed_dim
            )
        else:
            self.tactile_patch_embed = TactilePatchEmbed(
                tactile_dim=tactile_dim,
                num_tactile_patches=tactile_history,
                embed_dim=embed_dim
            )
        
        img_patches = self.rgb_patch_embed.img_patches
        tactile_patches = self.tactile_patch_embed.tactile_patches
        
        # Set up modality-specific positional embeddings
        self.pos_enc = nn.ParameterDict({
            'rgb': None,
            'tactile': None,
            'extra_tokens': None
        })

        self.pos_enc['rgb'] = nn.Parameter(torch.zeros(1, img_patches, embed_dim))
        self.pos_enc['tactile'] = nn.Parameter(torch.zeros(1, tactile_patches, embed_dim))
        
        # Set up transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        # Set up normalization layer
        self.norm = norm_layer(embed_dim)

        # Set up patch compression layer
        self.compress_patches = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim // 4, embed_dim // 12)
        )

        # Set up latent vector layer
        self.compress_layer = nn.Sequential(
            nn.Linear((img_patches * self.num_rgb + tactile_patches + num_extra_tokens) * embed_dim // 12, 640),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(640, output_dim)
        )
        
        # Initialize position encoding parameters
        for m in self.pos_enc.keys():
            if self.pos_enc[m] is not None:
                trunc_normal_(self.pos_enc[m])
    
    # Prepares observation tokens as input to attention layers
    def prepare_tokens(self, images, tactile):
        B, S, NC, w, h = images[0].shape

        # Tokenize observations
        images_patched = [self.rgb_patch_embed(img) for img in images]
        tactile_patched = self.tactile_patch_embed(tactile)

        # Apply modality-specific positional encodings
        images_patched = [img + self.pos_enc['rgb'] for img in images_patched]
        image_tokens = torch.cat(images_patched, dim=2)
        
        tactile_tokens = tactile_patched + self.pos_enc['tactile']

        # Group all tokens together
        x = torch.cat((image_tokens, tactile_tokens), dim=2)

        return x
    
    def forward(self, images: list, tactile: list):
        # DO NOT unsqueeze dimensions if sequence length is already there (for AIL)
        if images[0].ndim != 5:
            images = [img.unsqueeze(1) for img in images]

        tactile = tactile[0][...,-self.tactile_history:,:]

        if tactile.ndim != 4:
            tactile = tactile.unsqueeze(1)
        
        assert images[0].ndim == 5
        assert tactile.ndim == 4

        # Tokenize observations and add position encodings
        x = self.prepare_tokens(images, tactile)

        # Pass tokens through transformer layers
        for blk in self.blocks:
            x = blk(x)
        
        # Normalize transformer outputs
        x = self.norm(x)

        # Compress patches before flattening and concatenation
        x_out = self.compress_patches(x)

        # Flatten and transform output tokens into single latent vector
        B, S, patches, dim = x_out.size()
        x_out = x_out.view(B, S, -1)
        z = self.compress_layer(x_out).squeeze(1)
        
        return z
    
    # =============== For visualization purposes 
    def get_last_selfattn(self, images, tactile):
        images = [img.unsqueeze(1) for img in images]
        tactile = tactile[...,-self.tactile_history:,:].unsqueeze(1)
        x = self.prepare_tokens(images, tactile)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of last block
                return blk(x, return_attention=True)
    
    def visualize_attention(self, images, tactile):
        # make the images divisible by the patch size
        w = images[0].shape[-1] - images[0].shape[-1] % self.img_patch_size
        h = images[0].shape[-2] - images[0].shape[-2] % self.img_patch_size
        images = [img[...,:h,:w] for img in images]

        w_featmap = images[0].shape[-1] // self.img_patch_size
        h_featmap = images[0].shape[-2] // self.img_patch_size
        
        attentions = self.get_last_selfattn(images, tactile).squeeze(1)
        nh = attentions.shape[1] # number of heads

        # TODO: Aggregate attention across all tokens? (or try to implement heatmap from VTT)
        # keep only the output patch attention
        attentions = attentions[0, :, 0, num_extra_tokens:]

        # average the attentions across all heads
        attentions = torch.mean(attentions, dim=0)

        # separate modality-specific attentions
        img_patches = self.rgb_patch_embed.img_patches
        tactile_patches = self.tactile_patch_embed.tactile_patches
        img_attns = [attentions[i*img_patches:(i+1)*img_patches] for i in range(self.num_rgb)]
        tactile_attns = attentions[-tactile_patches:]

        # calculate modality-specific attention proportions
        img_proportion = torch.sum(attentions[:-tactile_patches]) / torch.sum(attentions)
        tactile_proportion = torch.sum(tactile_attns) / torch.sum(attentions)
        proportions = (img_proportion, tactile_proportion)

        # calculating attention proportions within modality-specific attention
        img_attns = [img_attn / torch.sum(img_attn) for img_attn in img_attns]
        tactile_attns = tactile_attns / torch.sum(tactile_attns)

        # turning image attentions into per-image heatmaps
        img_attns = [
            F.interpolate(
                img_attn.reshape(1, h_featmap, w_featmap).unsqueeze(0),
                scale_factor=self.img_patch_size,
                mode="bilinear"
            )[0] for img_attn in img_attns
        ]

        return img_attns, tactile_attns, proportions

    # ===============

    def output_shape(self, input_shape=None):
        return [self.output_dim]

class RGBPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=84,
        img_patch_size=14,
        in_channels=3,
        embed_dim=384,
    ):
        super().__init__()
        self.img_patches = int((img_size/img_patch_size)*(img_size/img_patch_size))
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.rgb_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=img_patch_size, stride=img_patch_size)
    
    def forward(self, image):
        # Input shape: batch, sequence, in_channels, H, W
        # Output shape: batch, sequence, patches, embed_dim
        B, S, C, H, W = image.shape
        image = image.view(B * S, C, H, W)
        patched_image = self.rgb_proj(image).flatten(2).transpose(1,2).view(B, S, -1, self.embed_dim)
        return patched_image

class TactilePatchEmbed(nn.Module):
    def __init__(
        self,
        tactile_dim=12,
        num_tactile_patches=32, # Should be the same as desired FT history length
        embed_dim=384,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tactile_patches = num_tactile_patches
        self.tactile_proj = nn.Linear(tactile_dim, embed_dim)

    def forward(self, tactile):
        # Input shape: batch, sequence, history length, tactile_dim
        # Output shape: batch, sequence, history length, embed_dim
        B, S, H, N = tactile.shape
        tactile = tactile.reshape((B*S*H, -1))
        patched_tactile = self.tactile_proj(tactile).view(B, S, self.tactile_patches, -1)
        return patched_tactile

# From original VTT paper (only takes in current force-torque reading)
class TactilePatchEmbedNoHistory(nn.Module):
    def __init__(
        self,
        tactile_dim=12,
        num_tactile_patches=4,
        embed_dim=384,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tactile_patches = num_tactile_patches
        self.tactile_proj = nn.Linear(tactile_dim, num_tactile_patches*embed_dim)
    
    def forward(self, tactile):
        # Input shape: batch, sequence, tactile_dim
        # Output shape: batch, sequence, num_tactile_patches, embed_dim
        tactile = tactile.squeeze(2)
        B, S, N = tactile.shape
        tactile = tactile.view(B*S, -1)
        patched_tactile = self.tactile_proj(tactile).view(B, S, self.tactile_patches, -1)
        return patched_tactile

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
    
    def forward(self, x, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5   # Default scaling factor of 1/sqrt(embed_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, S, N, D = x.shape
        qkv = self.qkv(x).reshape(B*S, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, S, N, D)
        attn = attn.view(B, S, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    @staticmethod
    def drop_path(x, drop_prob: float = 0.0, training: bool = False):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return DropPath.drop_path(x, self.drop_prob, self.training)
    
class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MLP = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Linear(hidden_features, out_features)
        )
    
    def forward(self, x):
        x = self.MLP(x)
        return x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
