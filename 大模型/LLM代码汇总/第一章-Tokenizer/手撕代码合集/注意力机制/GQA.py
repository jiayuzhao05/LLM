import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_groups == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_groups
        self.group_dim = self.head_dim * num_groups
        self.d_model = d_model        
        # Linear layers for Query, Key, and Value
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, self.group_dim)  # Shared Key per group
        self.wv = nn.Linear(d_model, self.group_dim)  # Shared Value per group        
        self.fc = nn.Linear(d_model, d_model)   
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)    
    def forward(self, x, mask=None):
        batch_size = x.size(0)      
        # Linear projections
        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, group_dim)
        v = self.wv(x)  # (batch_size, seq_len, group_dim)      
        # Split Query into heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, head_dim)        
        # Split Key and Value into groups and expand within the group
        k = k.view(batch_size, -1, self.num_groups, self.head_dim)
        v = v.view(batch_size, -1, self.num_groups, self.head_dim)        
        k = k.repeat(1, 1, 1, self.num_heads // self.num_groups)
        v = v.repeat(1, 1, 1, self.num_heads // self.num_groups)      
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)        
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_dim)      
        # Concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)       
        # Final linear layer
        output = self.fc(output)  # (batch_size, seq_len, d_model        
        return output, attention_weights