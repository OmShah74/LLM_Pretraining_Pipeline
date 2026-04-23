from __future__ import annotations

import torch

from src.data.tokenizer import BPETokenizer
from src.model.transformer import MoEDecoderLM
from src.utils.contracts import GenerationConfig


@torch.no_grad()
def generate_text(
    model: MoEDecoderLM,
    tokenizer: BPETokenizer,
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    for _ in range(config.max_new_tokens):
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :] / max(config.temperature, 1e-5)
        top_k = min(config.top_k, next_token_logits.size(-1))
        top_logits, top_indices = torch.topk(next_token_logits, k=top_k, dim=-1)
        probs = torch.softmax(top_logits, dim=-1)
        sampled = top_indices.gather(-1, torch.multinomial(probs, num_samples=1))
        input_ids = torch.cat([input_ids, sampled], dim=1)
        if sampled.item() == tokenizer.token_to_id[tokenizer.eos_token]:
            break
    return tokenizer.decode(input_ids[0].tolist())
