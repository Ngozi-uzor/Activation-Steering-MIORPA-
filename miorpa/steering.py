"""Forward-hook activation steering and batched generation.

The hook adds alpha * vector to the residual stream at one layer. Two
things matter here:

1. The prompt itself is left unsteered, except its final position, which
   seeds the first generated token. Steering every prompt token distorts
   how the model reads the question itself and breaks grammar rather than
   shifting the answer's content.

2. The hook has to handle both shapes a decoder block can return: a bare
   tensor, or a tuple whose first element is the hidden state.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from . import config as cfg
from .vectors import layer_module


class SteeringHook:
    """Adds alpha * vector to the residual stream at one layer.

    During generation, the first forward pass covers the whole prompt and
    every pass after that covers a single new token. This tracks which
    pass it's on so the prompt is left untouched when steer_prompt is False.
    """

    def __init__(self, vector: torch.Tensor, alpha: float, steer_prompt: bool = False):
        self.vector = vector
        self.alpha = alpha
        self.steer_prompt = steer_prompt
        self.is_first_pass = True

    def reset(self):
        self.is_first_pass = True

    def __call__(self, module, inputs, output):
        returns_tuple = isinstance(output, tuple)
        hidden = output[0] if returns_tuple else output

        if self.alpha == 0.0:
            self.is_first_pass = False
            return output

        delta = (self.alpha * self.vector).to(dtype=hidden.dtype, device=hidden.device)

        if self.is_first_pass and not self.steer_prompt:
            # Prompt pass: steer only the last position, since that's the
            # one that seeds the first generated token.
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + delta
        else:
            hidden = hidden + delta

        self.is_first_pass = False
        return (hidden,) + output[1:] if returns_tuple else hidden


@contextlib.contextmanager
def steering(model, layer_idx: int, vector, alpha: float, steer_prompt: bool | None = None):
    """Attach a steering hook for the duration of the block."""
    if steer_prompt is None:
        steer_prompt = cfg.STEER_PROMPT_TOKENS

    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    vector = vector.float()

    hook = SteeringHook(vector, alpha, steer_prompt)
    handle = layer_module(model, layer_idx).register_forward_hook(hook)
    try:
        yield hook
    finally:
        handle.remove()


def build_chat_prompts(tokenizer, questions, system: str | None = None):
    """Apply the model's chat template so instruct models behave properly."""
    prompts = []
    for question in questions:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": question}
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = (f"{system}\n\n" if system else "") + f"User: {question}\nAssistant:"
        prompts.append(prompt_text)
    return prompts


@torch.no_grad()
def generate(
    model,
    tokenizer,
    questions,
    device: str,
    layer_idx: int | None = None,
    vector=None,
    alpha: float = 0.0,
    batch_size: int | None = None,
    max_new_tokens: int | None = None,
    seed: int | None = None,
):
    """Generate answers, optionally steered. Returns a list of strings.

    Passing vector=None or alpha=0 gives the unsteered baseline.
    """
    batch_size = batch_size or cfg.GENERATION_BATCH_SIZE
    max_new_tokens = max_new_tokens or cfg.MAX_NEW_TOKENS
    seed = cfg.SEED if seed is None else seed

    prompts = build_chat_prompts(tokenizer, questions)
    answers = []

    should_steer = vector is not None and alpha != 0.0
    steering_context = (
        steering(model, layer_idx, vector, alpha)
        if should_steer
        else contextlib.nullcontext(None)
    )

    with steering_context as hook:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)

            if hook is not None:
                hook.reset()

            torch.manual_seed(seed + start)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=cfg.TEMPERATURE > 0,
                temperature=cfg.TEMPERATURE,
                top_p=cfg.TOP_P,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Strip the prompt back off; left padding makes it a fixed-width prefix.
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            answers.extend(text.strip() for text in decoded)

    return answers
