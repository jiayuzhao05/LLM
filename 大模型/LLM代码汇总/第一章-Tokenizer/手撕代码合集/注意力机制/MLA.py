# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        # 对应 query 压缩后的隐向量的维度 d'_c
        self.q_lora_rank = config.q_lora_rank
        # 对应$d_h^R$， 表示应用了rope的 queries 和 key 的一个 head 的维度。
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # 对应 key-value 压缩后的隐向量维度 d_c
        self.kv_lora_rank = config.kv_lora_rank
        # value 的一个注意力头的隐藏层为度
        self.v_head_dim = config.v_head_dim
        # 表示query和key的隐藏向量中应用rope部分的维度
        self.qk_nope_head_dim = config.qk_nope_head_dim
        # 每一个注意力头的维度应该是两部分只和
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True
                         
        # MLA 中对 Q 投影矩阵也做了一个低秩分解，对应生成 q_a_proj 和 q_b_proj 两个矩阵。
        # 其中 q_a_proj 大小为 [hidden_size, q_lora_rank] = [5120, 1536]，
        # 对应上面公式中的W^DQ
        self.q_a_proj = nn.Linear(
            self.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
        # q_b_proj 大小为 [q_lora_rank, num_heads * q_head_dim] = 
        # [q_lora_rank, num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)] = [1536, 128*(128+64)] = [1536, 24576] 
        # 对应上述公式中的W^UQ和W^QR合并后的大矩阵
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
                # 与Q向量类似，KV向量的生成也是先投影到一个低维的 compressed_kv 向量（对应c_t^{KV}）
                # 再升维展开。具体的代码涉及 kv_a_proj_with_mqa 和 kv_b_proj 两个参数矩阵。
                # 其中 kv_a_proj_with_mqa 大小为 [hidden_size， kv_lora_rank + qk_rope_head_dim]
                # = [5120, 512 + 64] = [5120, 576]，对应上述公式中的W^{DKV}和W^{KR}。
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
     # kv_b_proj 大小为 [kv_lora_rank， num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim)] 
     # = [512, 128*((128+64)-64+128)] = [512, 32768]，对应上述公式中的W^{UK}和W^{UV}。
     # 由于 W^{UK} 只涉及 non rope 的部分所以维度中把 qk_rope_head_dim 去掉了。
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
                         # 对应完整公式的第 47 行
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        # hidden_states对应公式中的h_t，的shape是(batch_size, seq_length,         
        # hidden_size)，其中 hidden_size 具体为 5120，假设batch_size和seq_length都为1
        bsz, q_len, _ = hidden_states.size()

        # 计算Q：对应完整公式中的 37-39 行，先降维再升维，好处是相比直接使用大小为 [5120, 24576] 的矩阵
       # [5120, 1536] * [1536, 24576] 这样的低秩分解在存储空间和计算量上都大幅度降低
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        # 切分 rope 和非 rope 部分，完整公式中 40 行反过来
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # 对应公式中的 41 和 43 行只是还没有加 rope
        # 一个优化的 MLA KVCache 实现只需要缓存这个 compressed_kv 就行
        # kv_a_proj_with_mqa shape 为[hidden_size， kv_lora_rank + qk_rope_head_dim]
                # = [5120, 512 + 64] = [5120, 576]
            # 所以compressed_kv的shape就是[1, 1, 576]
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # 对应完整公式的 44 行反过来
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # 这里的 k_pe 和 上面的 q_pe 要扔给 RoPE模块，所以需要重整下shape
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        # 对应公式中的 42 和 45 行，将 MLA 展开成标准 MHA 的形式
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
                         # 因为 kv_b_proj 打包了 W^{UK} 和 W^{UV} 把他们分离出来
        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # 获取key/value的序列长度
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # 给需要 rope 的部分加 rope
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # 更新和拼接历史 KVCache，可以看到这里存储的是展开后的 MHA KVCache
        # 其中 q_head_dim 等于 qk_nope_head_dim + qk_rope_head_dim
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        # Transformers库中标准的 KV Cache 更新代码
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 后续就是标准的多头自注意力计算了，为了篇幅，忽略这部分代码
        ...