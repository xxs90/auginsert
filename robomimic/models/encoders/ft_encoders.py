import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from torchvision import transforms
from robomimic.models.base_nets import Module
from robomimic.models.transformers import SelfAttentionBlock, CausalSelfAttention, PositionalEncoding

import warnings
import math

class CausalConvForceTorqueEncoder(Module):
    """
    Base class for stacked CausalConv1d layers.
    Used for force-torque history encoding

    Args:
        input_channel (int): Number of channels for inputs to this network
        activation (None or str): Per-layer activation to use. Defaults to "relu". Valid options are
            currently {relu, None} for no activation
        out_channels (list of int): Output channel size for each sequential Conv1d layer
        kernel_size (list of int): Kernel sizes for each sequential Conv1d layer
        stride (list of int): Stride sizes for each sequential Conv1d layer
        conv_kwargs (dict): additional nn.Conv1D args to use, in list form, where the ith element corresponds to the
            argument to be passed to the ith Conv1D layer.
            See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html for specific possible arguments.
    """
    def __init__(
        self,
        input_channel=12,
        activation="relu",
        out_channels=(16, 32, 64, 128),
        output_dim=32,
        kernel_size=(2, 2, 2, 2, 2),
        stride=(2, 2, 2, 2, 2), # NOTE: This architecture REQUIRES a history length of 32
        **conv_kwargs,
    ):
        super(CausalConv1dBase, self).__init__()

        # Get activation requested
        activation = nn.LeakyReLU # CONV_ACTIVATIONS[activation]

        # Add output dimension to out channels
        out_channels += (output_dim,)

        assert len(out_channels) == len(kernel_size) and len(out_channels) == len(stride)
        
        # Add layer kwargs
        conv_kwargs["out_channels"] = out_channels
        conv_kwargs["kernel_size"] = kernel_size
        conv_kwargs["stride"] = stride

        # Generate network
        self.n_layers = len(out_channels)
        layers = OrderedDict()
        for i in range(self.n_layers):
            layer_kwargs = {k: v[i] for k, v in conv_kwargs.items()}
            layers[f'conv{i}'] = CausalConv1d(
                in_channels=input_channel,
                **layer_kwargs,
            )
            if activation is not None:
                layers[f'act{i}'] = activation(0.1, inplace=True) # Slope for LeakyReLU
            input_channel = layer_kwargs["out_channels"]

        # Store network
        self.nets = nn.Sequential(layers)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        channels, length = input_shape
        for i in range(self.n_layers):
            net = getattr(self.nets, f"conv{i}")
            channels = net.out_channels
            length = int((length + 2 * net.padding[0] - net.dilation[0] * (net.kernel_size[0] - 1) - 1) / net.stride[0]) + 1 - net.padding[0]
        return [channels, length]

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x

class CausalSelfAttnForceTorqueEncoder(Module):
    '''
    Performs causal self-attention on a sequence on inputs
    '''
    
    def __init__(self, input_dim, seq_len=32, output_dim=64, d_model=32):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.z_dim = output_dim

        # 1: Linear projection of patches: ConvBlock / MLP?
        # self.linear_proj = ConvBlock(self.input_dim, self.d_model, kernel_size=1, stride=1, padding=0)
        self.linear_proj = nn.Linear(self.input_dim, self.d_model)

        # 2: Add on Positional Encoding (in this case position represents timestep): PositionalEncoding
        self.pos_enc = PositionalEncoding(self.d_model)

        # 3: Feed through Causal Self Attention module: SelfAttentionBlock
        self.attn = CausalSelfAttention(embed_dim=self.d_model, num_heads=1, context_length=self.seq_len)

        self.norm = nn.LayerNorm(self.d_model)

        # 4: Linear projection into single embedding: ConvBlock
        # self.embed = ConvBlock(self.seq_len * self.d_model, self.z_dim, kernel_size=1, stride=1, padding=0)
        self.embed = nn.Linear(self.seq_len * self.d_model, self.z_dim)
    
    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        assert torch.is_floating_point(timesteps), timesteps.dtype
        time_embeddings = self.pos_enc(timesteps)  # these are NOT fed into transformer, only added to the inputs.
        # compute how many modalities were combined into embeddings, replicate time embeddings that many times
        num_replicates = embeddings.shape[-1] // self.d_model
        time_embeddings = torch.cat([time_embeddings for _ in range(num_replicates)], -1)
        assert (
            embeddings.shape == time_embeddings.shape
        ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] == self.seq_len, f"Force-torque sequence length mismatch: input {x.shape[1]}, expected {self.seq_len}"

        x = self.linear_proj(x.view(batch_size * self.seq_len, self.input_dim)).view(batch_size, self.seq_len, self.d_model)
        pe = self.embed_timesteps(x)
        x = x + pe
        x = self.norm(x)
        x = self.attn(x)
        x = self.embed(x.reshape(batch_size, -1))
        return x

    def output_shape(self, input_shape):
        return [self.z_dim]

class TransformerForceTorqueEncoder(Module):
    '''
    A more complete transformer architecture involving causal self-attention on a sequence on inputs
    '''
    
    def __init__(self, input_dim, seq_len=32, output_dim=64, num_heads=4, d_model=128):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.z_dim = output_dim

        # 1: Linear projection of patches: Linear
        self.linear_proj = nn.Linear(self.input_dim, self.d_model)

        # 2: Add on Positional Encoding (in this case position represents timestep): PositionalEncoding
        self.pos_enc = PositionalEncoding(self.d_model)

        # 3: Feed through sequence of Self Attention modules 
        self.blocks = nn.Sequential(*[
            SelfAttentionBlock(
                embed_dim=self.d_model,
                num_heads=num_heads,
                context_length=self.seq_len
            ) for _ in range(6)
        ])

        # 4: Linear projection into single embedding: Linear
        self.embed = nn.Linear(self.seq_len * self.d_model, self.z_dim)
    
    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        assert torch.is_floating_point(timesteps), timesteps.dtype
        time_embeddings = self.pos_enc(timesteps)  # these are NOT fed into transformer, only added to the inputs.
        # compute how many modalities were combined into embeddings, replicate time embeddings that many times
        num_replicates = embeddings.shape[-1] // self.d_model
        time_embeddings = torch.cat([time_embeddings for _ in range(num_replicates)], -1)
        assert (
            embeddings.shape == time_embeddings.shape
        ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings
    
    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] == self.seq_len, f"Force-torque sequence length mismatch: input {x.shape[1]}, expected {self.seq_len}"
        x = self.linear_proj(x.view(batch_size * self.seq_len, self.input_dim)).view(batch_size, self.seq_len, self.d_model)
        pe = self.embed_timesteps(x)
        x = x + pe
        x = self.blocks(x)
        x = self.embed(x.reshape(batch_size, -1))
        return x

    def output_shape(self, input_shape):
        return [self.z_dim]
