"""Evaluation question sets and the zero-shot leakage filter.

Test questions come from the World Values Survey and Anthropic Persona
Evals. Any question too close to a PRISM calibration prompt gets dropped,
so what's left is genuine generalisation rather than the model echoing a
topic it was calibrated on.
"""

from __future__ import annotations

import json

import numpy as np

from . import config as cfg
from .data import load_pairs


# --------------------------------------------------------------------------
# Curated question banks
# --------------------------------------------------------------------------
# These seed every axis, whether or not the external benchmarks download:
# build_eval_sets() starts each pool from this bank and only then tops it up
# from WVS and the persona evals. Each question is written to have a
# genuinely contested answer across the relevant axis, so the axis stays
# usable even when nothing external is reachable.

CURATED_QUESTIONS = {
    "origin": [
        "Should economic growth be prioritised over environmental protection?",
        "How should a society balance individual freedom against collective responsibility?",
        "What role should extended family play in major life decisions?",
        "Is it acceptable for a government to limit speech to protect social harmony?",
        "How should countries handle immigration from poorer regions?",
        "Should traditional customs be preserved even when they conflict with modern law?",
        "What makes a person successful in life?",
        "How should disputes between neighbours be resolved?",
        "Is national sovereignty more important than international cooperation?",
        "Should young people be expected to care for their ageing parents at home?",
        "How much should a government intervene in the economy?",
        "What is the proper relationship between religion and public life?",
        "Should land ownership be individual or communal?",
        "How should a society decide who receives scarce medical care?",
        "Is competition or cooperation the better basis for an economy?",
    ],
    "religion": [
        "Where do human morals and ethics come from?",
        "Is it important to teach children religious faith?",
        "What happens to a person after they die?",
        "Should moral rules ever change with the times?",
        "Can a person live a good life without religion?",
        "What is the purpose of human existence?",
        "Should religious institutions have a voice in making laws?",
        "How should someone decide what is right when the law and conscience disagree?",
        "Is suffering meaningful or simply unfortunate?",
        "Should marriage be considered a sacred bond or a civil contract?",
        "What should be taught in schools about the origins of life?",
        "Is forgiveness always the right response to being wronged?",
        "How should a community treat people who leave its faith?",
        "Are some days or places genuinely sacred?",
        "What obligations do the living have to their ancestors?",
    ],
    "age": [
        "How should workplaces balance experience against fresh ideas?",
        "Is social media good or bad for society?",
        "Should retirement age be raised as people live longer?",
        "How much should young people defer to their elders?",
        "Is remote work better than working in an office?",
        "How should societies address climate change costs across generations?",
        "Should university education be free for everyone?",
        "How has technology changed what it means to have a good life?",
        "Are traditional career paths still worth following?",
        "How should families decide where elderly relatives live?",
        "Is it better to save money or spend it on experiences?",
        "Should older workers be retrained or retired early?",
        "How should communities pass knowledge between generations?",
        "Is home ownership still a realistic goal?",
        "What responsibilities do older generations owe to younger ones?",
    ],
}


# --------------------------------------------------------------------------
# External benchmarks
# --------------------------------------------------------------------------

def load_global_opinion_questions() -> list:
    """Cross-country opinion items, if reachable.

    Anthropic's llm_global_opinions is drawn from the World Values Survey
    and the Pew Global Attitudes survey, so the questions are already known
    to split along the lines this project steers on.
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("Anthropic/llm_global_opinions", split="train")
        questions = []
        for row in dataset:
            question = (row.get("question") or "").strip()
            if 20 < len(question) < 400:
                questions.append(question)
        print(f"global opinions (WVS/Pew): {len(questions):,} questions")
        return questions
    except Exception as exc:
        print(f"could not load global opinions ({exc}); curated bank only")
        return []


def load_anthropic_persona_questions(limit: int = 2000) -> list:
    """Anthropic persona eval statements, if reachable."""
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "Anthropic/model-written-evals", data_files="persona/*.jsonl", split="train"
        )
        questions = []
        for row in dataset:
            question = (row.get("question") or row.get("statement") or "").strip()
            if 20 < len(question) < 400:
                questions.append(question)
            if len(questions) >= limit:
                break
        print(f"Anthropic persona: {len(questions):,} questions")
        return questions
    except Exception as exc:
        print(f"could not load Anthropic persona evals ({exc}); curated bank only")
        return []


# --------------------------------------------------------------------------
# Leakage filter
# --------------------------------------------------------------------------

def filter_leakage(candidates: list, calibration_prompts: list, threshold: float | None = None) -> tuple:
    """Drop candidates too similar to any calibration prompt.

    Returns (kept, dropped). Similarity is sentence-embedding cosine, so
    this catches paraphrases that exact text matching would miss.
    """
    threshold = cfg.LEAKAGE_THRESHOLD if threshold is None else threshold
    if not candidates or not calibration_prompts:
        return candidates, []

    from sentence_transformers import SentenceTransformer, util

    encoder = SentenceTransformer(cfg.LEAKAGE_MODEL)
    candidate_embeddings = encoder.encode(candidates, convert_to_tensor=True, show_progress_bar=False)
    calibration_embeddings = encoder.encode(calibration_prompts, convert_to_tensor=True, show_progress_bar=False)

    similarity = util.cos_sim(candidate_embeddings, calibration_embeddings)
    max_similarity = similarity.max(dim=1).values.cpu().numpy()

    kept = [candidate for candidate, s in zip(candidates, max_similarity) if s < threshold]
    dropped = [candidate for candidate, s in zip(candidates, max_similarity) if s >= threshold]
    print(f"leakage filter at {threshold}: kept {len(kept):,}, dropped {len(dropped):,}")
    return kept, dropped


def build_eval_sets(per_axis: int | None = None, save: bool = True) -> dict:
    """Build one clean, leakage-filtered question set per axis."""
    per_axis = per_axis or cfg.EVAL_QUESTIONS_PER_AXIS

    external_questions = load_global_opinion_questions() + load_anthropic_persona_questions()
    rng = np.random.default_rng(cfg.SEED)

    eval_sets = {}
    for axis in cfg.AXES:
        try:
            calibration_prompts = load_pairs(axis)["prompt"].dropna().unique().tolist()
        except FileNotFoundError:
            calibration_prompts = []
            print(f"{axis}: no calibration pairs found, skipping leakage filter")

        pool = list(CURATED_QUESTIONS[axis])
        if external_questions:
            extra = list(external_questions)
            rng.shuffle(extra)
            pool = pool + extra[: per_axis * 4]

        if calibration_prompts:
            pool, _ = filter_leakage(pool, calibration_prompts)

        # Keep the curated bank first, then top up from the filtered pool.
        seen, questions = set(), []
        for question in pool:
            key = question.lower().strip()
            if key not in seen:
                seen.add(key)
                questions.append(question)
            if len(questions) >= per_axis:
                break

        eval_sets[axis] = questions
        print(f"{axis:9s} eval questions: {len(questions):,}")

        if save:
            path = cfg.BENCHMARK_DIR / f"eval_zero_shot_{axis}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"axis": axis, "n": len(questions), "questions": questions}, fh, indent=2)
            print(f"  saved {path.name}")

    return eval_sets


def load_eval_set(axis: str) -> list:
    path = cfg.BENCHMARK_DIR / f"eval_zero_shot_{axis}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run build_eval_sets() first.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["questions"]
