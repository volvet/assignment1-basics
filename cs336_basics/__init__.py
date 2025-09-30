import importlib.metadata

from .data_loader import DataLoader
from .linear import Linear
from .embedding import Embedding
from .rms_norm import RMSNorm
from .position_wise_feed_forward import (
    PositionWiseFeedForward,
    silu,
)
from .rotary_position_embedding import RotaryPositionEmbedding
from .attention import (
    softmax,
    ScaledDotProductAttention,
    MultiHeadSelfAttention,
)
from .transformer import (
    TransformerBlock,
    TransformerLM,
)

from .adamw import ADAMW

from .utils import (
    cross_entropy_loss,
    clip_gradients,
)

__version__ = importlib.metadata.version("cs336_basics")
