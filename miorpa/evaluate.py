"""The two fully-automatic metrics: BERTScore and perplexity.

    bertscore   - is the answer still about the question that was asked
    perplexity  - is the answer still fluent English

They measure different things on purpose. An answer can be fluent but off
topic, or on topic but garbled, and one blended "coherence" number would
hide both cases.

Pluralism itself is not scored here, that needs a judge. See judge.py for
the pairwise LLM-as-judge system.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from . import config as cfg


# ==========================================================================
# BERTScore - question/answer relevance
# ==========================================================================

_bert_scorer = None


def _get_bert_scorer():
    global _bert_scorer
    if _bert_scorer is None:
        from bert_score import BERTScorer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _bert_scorer = BERTScorer(
                model_type=cfg.BERTSCORE_MODEL,
                lang="en",
                rescale_with_baseline=True,
                device=device,
            )
        except Exception as exc:
            print(f"BERTScore falling back to {cfg.BERTSCORE_FALLBACK} ({exc})")
            _bert_scorer = BERTScorer(
                model_type=cfg.BERTSCORE_FALLBACK,
                lang="en",
                rescale_with_baseline=True,
                device=device,
            )

        # Some tokenizers (DeBERTa included) don't set a real max length, so
        # transformers falls back to a sentinel meaning "unbounded", around
        # 10^30. bert_score passes that straight into the fast tokenizer's
        # truncation setup, which expects an ordinary integer and overflows.
        # Clamp it to the standard length for this model family before any
        # scoring happens.
        tokenizer = _bert_scorer._tokenizer
        if tokenizer.model_max_length > 100_000:
            tokenizer.model_max_length = 512
    return _bert_scorer


def bertscore_relevance(questions, answers, batch_size: int = 64) -> np.ndarray:
    """F1 BERTScore of each answer against its own question.

    High means the answer is semantically about the question. This is a
    relevance check, not a quality check: a fluent answer to a different
    question scores low, which is exactly what should be caught here.
    """
    scorer = _get_bert_scorer()

    clean_questions, clean_answers, kept_indices = [], [], []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        if answer and str(answer).strip():
            clean_questions.append(str(question))
            clean_answers.append(str(answer))
            kept_indices.append(i)

    scores = np.full(len(answers), np.nan, dtype=np.float32)
    if not clean_answers:
        return scores

    _, _, f1 = scorer.score(clean_answers, clean_questions, batch_size=batch_size)
    for index, value in zip(kept_indices, f1.tolist()):
        scores[index] = value
    return scores


# ==========================================================================
# Perplexity - fluency
# ==========================================================================

_perplexity_model = None
_perplexity_tokenizer = None


def _get_perplexity_scorer():
    """One fixed scorer for every model under test.

    Scoring a model's own output with itself makes the numbers incomparable
    across models: a 3B always finds its own text more predictable than a
    135M finds its own. A single external scorer keeps everything on one scale.
    """
    global _perplexity_model, _perplexity_tokenizer
    if _perplexity_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _perplexity_tokenizer = AutoTokenizer.from_pretrained(cfg.PERPLEXITY_SCORER)
        if _perplexity_tokenizer.pad_token is None:
            _perplexity_tokenizer.pad_token = _perplexity_tokenizer.eos_token
        _perplexity_model = AutoModelForCausalLM.from_pretrained(
            cfg.PERPLEXITY_SCORER,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        _perplexity_model.eval()
    return _perplexity_model, _perplexity_tokenizer


@torch.no_grad()
def perplexity(texts, max_length: int = 256) -> np.ndarray:
    """Token-level perplexity under the fixed scorer. Lower is more fluent.

    Computed one text at a time so padding never contributes to the loss,
    which is the usual source of wrong perplexity numbers in batched code.
    """
    model, tokenizer = _get_perplexity_scorer()
    device = next(model.parameters()).device

    scores = np.full(len(texts), np.nan, dtype=np.float32)
    for i, text in enumerate(texts):
        if not text or not str(text).strip():
            continue
        encoded = tokenizer(
            str(text), return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        if encoded["input_ids"].shape[1] < 2:
            continue
        loss = model(**encoded, labels=encoded["input_ids"]).loss.item()
        if math.isfinite(loss):
            scores[i] = float(math.exp(min(loss, 20.0)))
    return scores


# ==========================================================================
# Scoring
# ==========================================================================

def score_dataframe(generations: pd.DataFrame) -> pd.DataFrame:
    """Attach BERTScore and perplexity to a generations table.

    Expects columns: question, response, plus whatever grouping columns.
    """
    scored = generations.copy()
    scored["response"] = scored["response"].fillna("").astype(str)

    print(f"bertscore on {len(scored):,} rows")
    scored["bertscore"] = bertscore_relevance(scored["question"].tolist(), scored["response"].tolist())

    print(f"perplexity on {len(scored):,} rows")
    scored["perplexity"] = perplexity(scored["response"].tolist())

    scored["n_words"] = scored["response"].str.split().str.len().fillna(0).astype(int)
    return scored


GROUP_COLUMNS = ("model", "axis", "method", "alpha", "condition")


def _group_columns(df: pd.DataFrame) -> list:
    return [c for c in GROUP_COLUMNS if c in df.columns]


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse scored generations into the table that goes in the paper.

    BERTScore is averaged, perplexity is taken as a median. That difference
    is deliberate: a single broken generation can score in the thousands and
    would drag a mean perplexity with it, while BERTScore is bounded and
    averages cleanly. The `perplexity` column below is therefore a median,
    unlike every other aggregate here.
    """
    group_columns = _group_columns(scored) or ["condition"]

    summary = scored.groupby(group_columns, dropna=False).agg(
        n=("response", "size"),
        bertscore=("bertscore", "mean"),
        bertscore_sd=("bertscore", "std"),
        perplexity=("perplexity", "median"),
        n_words=("n_words", "mean"),
        empty_rate=("response", lambda s: float((s.str.strip() == "").mean())),
    )
    return summary.reset_index().sort_values(group_columns)
