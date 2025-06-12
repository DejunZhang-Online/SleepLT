from functools import partial
from typing import Callable, Tuple
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)

class Seq_level_Encoder(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.RMSNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = MultiHeadLatentAttention_DeepSeek_FAN(
            d_model=128,  # Model dimension
            num_head=4,  # Number of attention heads
            d_embed=128,  # Embedding dimension
            d_c=32,  # KV compression dimension
            d_c1=32,  # Query compression dimension
            d_rotate=32,  # Rotary embedding dimension
            bias=False
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, with_gate = True):
        super(FANLayer, self).__init__()
        print("FANLayer input_dim:{}, output_dim:{}".format(input_dim, output_dim))
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2), bias=bias)
        self.activation = nn.GELU()        
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate*torch.cos(p), gate*torch.sin(p), (1-gate)*g), dim=-1)
        return output

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Validate input dimensions
    assert xq.shape[-1] == xk.shape[-1], "Query and Key must have same embedding dimension"
    assert xq.shape[-1] % 2 == 0, "Embedding dimension must be even"

    # Get sequence lengths
    q_len = xq.shape[1]
    k_len = xk.shape[1]

    # Use appropriate part of freqs_cis for each sequence
    q_freqs = freqs_cis[:q_len]
    k_freqs = freqs_cis[:k_len]

    # Apply rotary embeddings separately
    # split last dimention to [xq.shape[:-1]/2, 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs for each
    q_freqs = reshape_for_broadcast(q_freqs, xq_)
    k_freqs = reshape_for_broadcast(k_freqs, xk_)

    # Works for both [bsz, seqlen, n_heads*head_dim] and [bsz, seqlen, n_heads, head_dim]
    xq_out = torch.view_as_real(xq_ * q_freqs).flatten(xq.ndim - 1)
    xk_out = torch.view_as_real(xk_ * k_freqs).flatten(xk.ndim - 1)

    return xq_out.type_as(xq), xk_out.type_as(xk)    

class MultiHeadLatentAttention_DeepSeek_FAN(nn.Module):
    """
        Multi-Head Latent Attention(MLA) Module As in DeepSeek_V2 pape
        Key innovation from standard MHA:
             1. Low-Rank Key-Value Joint Compression 
             2. Decoupled Rotary Position Embedding

        Cite:
            [1] DeepSeek-AI, Liu A, Feng B, et al. DeepSeek-V3 technical report[Z]. arXiv, 2025(2025-02-18).
            https://huggingface.co/bird-of-paradise/deepseek-mla
             
    Args:
        d_model:  Total dimension of the model.
        num_head: Number of attention heads.
        d_embed:  Embedding dimension
        d_c:      K/V compression dimension
        d_c1:     Q compression dimension
        d_rotate: Dimension for Rotary Position Embedding
        dropout:  Dropout rate for attention scores.
        bias:     Whether to include bias in linear projections.

        d_head:   Inferred from d_model//num_head

    Inputs:
        sequence: input sequence for self-attention and the query for cross-attention
        key_value_state: input for the key, values for cross-attention
    """

    def __init__(
            self,
            d_model,  # Infer d_head from d_model
            num_head,
            d_embed,
            d_c,
            d_c1,
            d_rotate,
            dropout=0.1,
            bias=True,
            max_batch_size=32,  # For KV cache sizing
            max_seq_len=200  # For KV cache sizing
    ):
        super().__init__()

        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        assert d_c < d_embed, "Compression dim should be smaller than embedding dim"
        assert d_c1 < d_embed, "Query compression dim should be smaller than embedding dim"

        self.d_model = d_model
        self.num_head = num_head
        # Verify dimensions match up
        assert d_model % num_head == 0, f"d_model ({d_model}) must be divisible by num_head ({num_head})"
        self.d_head = d_model // num_head
        self.d_embed = d_embed
        self.d_c = d_c
        self.d_c1 = d_c1
        self.d_rotate = d_rotate
        self.dropout_rate = dropout  # Store dropout rate separately

        # Linear down-projection(compression) transformations
        self.DKV_proj = FANLayer(d_embed, d_c, bias=bias)
        self.DQ_proj = FANLayer(d_embed, d_c1, bias=bias)

        # linear up-projection transformations
        self.UQ_proj = FANLayer(d_c1, d_model, bias=bias)
        self.UK_proj = FANLayer(d_c, d_model, bias=bias)
        self.UV_proj = FANLayer(d_c, d_model, bias=bias)

        # Linear RoPE-projection
        self.RQ_proj = nn.Linear(d_c1, num_head * d_rotate, bias=bias)
        self.RK_proj = nn.Linear(d_embed, d_rotate, bias=bias)

        # linear output transformations
        self.output_proj = FANLayer(d_model, d_model, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Initiialize scaler
        self.scaler = float(1.0 / math.sqrt(self.d_head + d_rotate))  # Store as float in initialization

        # Initialize C_KV and R_K cache for inference
        self.cache_kv = torch.zeros(
            (max_batch_size, max_seq_len, d_c)
        )
        self.cache_rk = torch.zeros(
            (max_batch_size, max_seq_len, d_rotate)
        )

        # Initialize freqs_cis for RoPE
        self.freqs_cis = precompute_freqs_cis(
            d_rotate, max_seq_len * 2
        )

    def forward(
            self,
            sequence,
            key_value_states=None,
            att_mask=None,
            use_cache=False,
            start_pos: int = 0
    ):

        """
        Forward pass supporting both standard attention and cached inference
        Input shape: [batch_size, seq_len, d_model=num_head * d_head]
        Args:
            sequence: Input sequence [batch_size, seq_len, d_model]
            key_value_states: Optional states for cross-attention
            att_mask: Optional attention mask
            use_cache: Whether to use KV caching (for inference)
            start_pos: Position in sequence when using KV cache
        """
        batch_size, seq_len, model_dim = sequence.size()
        # prepare for RoPE
        self.freqs_cis = self.freqs_cis.to(sequence.device)
        freqs_cis = self.freqs_cis[start_pos:]

        # Check only critical input dimensions
        assert model_dim == self.d_model, f"Input dimension {model_dim} doesn't match model dimension {self.d_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.d_model, \
                f"Cross attention key/value dimension {key_value_states.size(-1)} doesn't match model dimension {self.d_model}"

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # Determine kv_seq_len early
        kv_seq_len = key_value_states.size(1) if is_cross_attention else seq_len

        # Linear projections and reshape for multi-head, in the order of Q, K/V
        # Down and up projection for query
        C_Q = self.DQ_proj(sequence)  # [batch_size, seq_len, d_c1]
        Q_state = self.UQ_proj(C_Q)  # [batch_size, seq_len, d_model]
        # Linear projection for query RoPE pathway
        Q_rotate = self.RQ_proj(C_Q)  # [batch_size, seq_len, num_head*d_rotate]

        if use_cache:
            # Equation (41) in DeepSeek-v2 paper: cache c^{KV}_t
            self.cache_kv = self.cache_kv.to(sequence.device)

            # Get current compressed KV states
            current_kv = self.DKV_proj(
                key_value_states if is_cross_attention else sequence)  # [batch_size, kv_seq_len, d_c]
            # Update cache using kv_seq_len instead of seq_len
            self.cache_kv[:batch_size, start_pos:start_pos + kv_seq_len] = current_kv
            # Use cached compressed KV up to current position
            C_KV = self.cache_kv[:batch_size, :start_pos + kv_seq_len]

            # Equation (43) in DeepSeek-v2 paper: cache the RoPE pathwway for shared key k^R_t
            assert self.cache_rk.size(-1) == self.d_rotate, "RoPE cache dimension mismatch"
            self.cache_rk = self.cache_rk.to(sequence.device)
            # Get current RoPE key
            current_K_rotate = self.RK_proj(
                key_value_states if is_cross_attention else sequence)  # [batch_size, kv_seq_len, d_rotate]
            # Update cache using kv_seq_len instead of seq_len
            self.cache_rk[:batch_size, start_pos:start_pos + kv_seq_len] = current_K_rotate
            # Use cached RoPE key up to current position
            K_rotate = self.cache_rk[:batch_size, :start_pos + kv_seq_len]  # [batch_size, cached_len, d_rotate]

            """handling attention mask"""
            if att_mask is not None:
                # Get the original mask shape
                mask_size = att_mask.size(-1)
                cached_len = start_pos + kv_seq_len  # cached key_len, including previous key
                assert C_KV.size(1) == cached_len, \
                    f"Cached key/value length {C_KV.size(1)} doesn't match theoretical length {cached_len}"

                # Create new mask matching attention matrix shape
                extended_mask = torch.zeros(
                    (batch_size, 1, seq_len, cached_len),  # [batch, head, query_len, key_len]
                    device=att_mask.device,
                    dtype=att_mask.dtype
                )

                # Fill in the mask appropriately - we need to be careful about the causality here
                # For each query position, it should only attend to cached positions up to that point
                for i in range(seq_len):
                    extended_mask[:, :, i, :(start_pos + i + 1)] = 0  # Can attend
                    extended_mask[:, :, i, (start_pos + i + 1):] = float('-inf')  # Cannot attend

                att_mask = extended_mask
        else:
            # Compression projection for C_KV
            C_KV = self.DKV_proj(key_value_states if is_cross_attention else sequence)  # [batch_size, kv_seq_len, d_c]\
            # RoPE pathway for *shared* key
            K_rotate = self.RK_proj(key_value_states if is_cross_attention else sequence)

        # Up projection for key and value
        K_state = self.UK_proj(C_KV)  # [batch_size, kv_seq_len/cached_len, d_model]
        V_state = self.UV_proj(C_KV)  # [batch_size, kv_seq_len/cached_len, d_model]

        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head)

        # After getting K_state from projection, get its actual sequence length
        actual_kv_len = K_state.size(1)  # kv_seq_len or start_pos + kv_seq_len
        # in cross-attention, key/value sequence length might be different from query sequence length
        # Use actual_kv_len instead of kv_seq_len for reshaping
        K_state = K_state.view(batch_size, actual_kv_len, self.num_head, self.d_head)
        V_state = V_state.view(batch_size, actual_kv_len, self.num_head, self.d_head)

        # Apply RoPE to query and shared key
        Q_rotate = Q_rotate.view(batch_size, seq_len, self.num_head, self.d_rotate)
        K_rotate = K_rotate.unsqueeze(2).expand(-1, -1, self.num_head, -1)  # [batch, cached_len, num_head, d_rotate]
        Q_rotate, K_rotate = apply_rotary_emb(Q_rotate, K_rotate, freqs_cis=freqs_cis)

        # Concatenate along head dimension
        Q_state = torch.cat([Q_state, Q_rotate], dim=-1)  # [batch_size, seq_len, num_head, d_head + d_rotate]
        K_state = torch.cat([K_state, K_rotate], dim=-1)  # [batch_size, actual_kv_len, num_head, d_head + d_rotate]

        # Scale Q by 1/sqrt(d_k)
        Q_state = Q_state * self.scaler
        Q_state = Q_state.transpose(1, 2)  # [batch_size, num_head, seq_len, head_dim]
        K_state = K_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]
        V_state = V_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]

        # Compute attention matrix: QK^T
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1, -2))

        # apply attention mask to attention matrix
        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask

        # apply softmax to the last dimension to get the attention score: softmax(QK^T)
        att_score = F.softmax(self.att_matrix, dim=-1)
        self.att_score = att_score  # Store attention score for visualization
        
        # apply drop out to attention score
        att_score = self.dropout(att_score)

        # get final output: softmax(QK^T)V
        att_output = torch.matmul(att_score, V_state)
    
        assert att_output.size(0) == batch_size, "Batch size mismatch"
        assert att_output.size(2) == seq_len, "Output sequence length should match query sequence length"

        # concatinate all attention heads
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_head * self.d_head)

        

        # final linear transformation to the concatenated output
        att_output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.d_model), \
            f"Final output shape {att_output.size()} incorrect"

        return att_output