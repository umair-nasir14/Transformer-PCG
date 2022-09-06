import torch
from torch import distributions
from torch import nn

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base







class TransformerBlock(nn.Module):
    """An ImageGPT Transformer block."""

    def __init__(self, n_channels, n_attention_heads):
        """Initializes a new TransformerBlock instance.

        Args:
            n_channels: The number of input and output channels.
            n_attention_heads: The number of attention heads to use.
        """
        super().__init__()
        self._ln1 = pg_nn.NCHWLayerNorm(n_channels)
        self._ln2 = pg_nn.NCHWLayerNorm(n_channels)
        self._attn = pg_nn.CausalAttention(
            in_channels=n_channels,
            n_heads=n_attention_heads,
            embed_channels=n_channels,
            out_channels=n_channels,
        )
        self._out = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels, out_channels=4 * n_channels, kernel_size=1
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=4 * n_channels, out_channels=n_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        x = x + self._attn(self._ln1(x))
        return x + self._out(self._ln2(x))

class ImageGPT(base.AutoregressiveModel):
    """The ImageGPT Model.

    Unlike [1], our implementation operates over image inputs, instead of
    embeddings. Furthermore, we implement skip connections from each block to the
    output. We find that this makes training a lot more stable and allows for much
    faster convergence.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        in_size_0=28,
        in_size_1=28,
        n_transformer_blocks=8,
        n_attention_heads=4,
        n_embedding_channels=16,
        sample_fn=None,
    ):
        """Initializes a new ImageGPT instance.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            in_size: Size of the input images. Used to create positional encodings.
            n_transformer_blocks: Number of TransformerBlocks to use.
            n_attention_heads: Number of attention heads to use.
            n_embedding_channels: Number of attention embedding channels to use.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._pos = nn.Parameter(torch.zeros(1, in_channels, in_size_0, in_size_1))
        self._input_1 = pg_nn.CausalConv2d(
            mask_center=True,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
        )
        self._input_2 = pg_nn.CausalConv2d(
            mask_center=True,
            in_channels=in_channels,
            out_channels=n_embedding_channels,
            kernel_size=3,
            padding=1,
        )

        self._transformer = nn.ModuleList(
            TransformerBlock(
                n_channels=n_embedding_channels, n_attention_heads=n_attention_heads
            )
            for _ in range(n_transformer_blocks)
        )
        self._ln = pg_nn.NCHWLayerNorm(n_embedding_channels)
        self._out_1 = nn.Conv2d(
            in_channels=n_embedding_channels, out_channels=out_channels, kernel_size=1
        )
        self._out_2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        #x = self._input_1(x + self._pos)
        #x = self._input_1(x + self._pos)
        #x = self._input_1(x + self._pos)
        #x = self._input_1(x + self._pos)
        x = self._input_2(x + self._pos)
        for block in self._transformer:
            x = x + block(x)
        #x = self._out_1(x)
        #x = self._out_1(self._ln(x))
        return self._out_1(self._ln(x))