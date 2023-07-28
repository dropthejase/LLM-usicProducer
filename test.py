from fast_transformers.masking import TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder

print(TransformerEncoderBuilder.from_kwargs(
                n_layers=1,
                n_heads=8,
                query_dimensions=512//8,
                value_dimensions=512//8,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="causal-linear",
            ).get())