def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if return_outputs:
        raise ValueError("The GRPOTrainer does not support returning outputs")
    # Compute the per-token log probabilities for the model

    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    # Compute the KL divergence between the model and the reference model
    ref_per_token_logps = inputs["ref_per_token_logps"]
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # x - x.detach() allows for preserving gradients from x
    advantages = inputs["advantages"]
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - self.beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Log the metrics
    completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
    self._metrics["completion_length"].append(completion_length)

    mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    return loss
def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
    ).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids[:, -logits_to_keep:]):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
    device = self.accelerator.device
    prompts = [x["prompt"] for x in inputs]
    prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
    prompt_inputs = self.processing_class(
        prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
    )
    prompt_inputs = super()._prepare_inputs(prompt_inputs)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    if self.max_prompt_length is not None:
        prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        
    # Regular generation path
    with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
        prompt_completion_ids = unwrapped_model.generate(
            prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
        )

    # Compute prompt length and extract completion ids
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]
    prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

    # Mask everything after the first EOS token
    is_eos = completion_ids == self.processing_class.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    # Concatenate prompt_mask with completion_mask for logit computation
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

    logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    with torch.inference_mode():
        ref_per_token_logps = self._get_per_token_logps(
            self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
        )
    # Decode the generated completions
    completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    if is_conversational(inputs[0]):
        completions = [[{"role": "assistant", "content": completion}] for completion in completions]

    # Compute the rewards
    prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]  # repeat prompts

    rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
    for i, (reward_func, reward_processing_class) in enumerate(
        zip(self.reward_funcs, self.reward_processing_classes)
    ):
        if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
            if is_conversational(inputs[0]):
                messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
            else:
                texts = [p + c for p, c in zip(prompts, completions)]
            reward_inputs = reward_processing_class(
                texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
            )
            reward_inputs = super()._prepare_inputs(reward_inputs)
            with torch.inference_mode():
                rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
        else:
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

    # Sum the rewards from all reward functions
    rewards = rewards_per_func.sum(dim=1)

    # Compute grouped-wise rewards
    mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

    # Normalize the rewards to compute the advantages
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
   
    return {"prompt_ids": prompt_ids,"prompt_mask": prompt_mask,"completion_ids": completion_ids,"completion_mask": completion_mask,"ref_per_token_logps": ref_per_token_logps,
        "advantages": advantages,
    }