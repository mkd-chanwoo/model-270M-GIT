import torch
import math
import torch.nn.functional as F

class model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.gpt_neox = Transformer(config)
        self.embed_out = torch.nn.Linear(self.hidden_size, self.vocab_size, bias=config.use_bias)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.gpt_neox(input_ids, attention_mask)
        logits = self.embed_out(hidden_states)
        return logits

class Transformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.layernorm_epsilon = config.layer_norm_eps   

        self.embed_in = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = torch.nn.ModuleList([TransformerBlock(config) for _ in range(self.num_hidden_layers)])
        self.final_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_in(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
    
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states
        

class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        
        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.num_attention_heads * self.head_dim, bias=config.use_bias)
        self.dense = torch.nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.use_bias)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.max_position_embeddings, self.max_position_embeddings, dtype=torch.bool)).view(1, 1, self.max_position_embeddings, self.max_position_embeddings),
            persistent=False,
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_length, self.num_attention_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = self.bias[:, :, :seq_length, :seq_length]
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.num_attention_heads * self.head_dim)
        output = self.dense(attn_output)

        return output

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense_h_to_4h = torch.nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.use_bias)
        self.dense_4h_to_h = torch.nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.use_bias)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states
