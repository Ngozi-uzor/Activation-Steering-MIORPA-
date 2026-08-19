"""PRISM loading, demographic mapping, and calibration pair extraction.

Pipeline:
    load_prism()              -> survey + conversations + utterances tables
    map_demographics()        -> user_id -> pole, for each axis
    build_rated_responses()   -> flat table of (prompt, response, rater, score)
    label_balanced_one_sided()-> tags each liked response balanced/one-sided
    extract_pairs()           -> matches a balanced response to a one-sided one

"Balanced" means a topic got a highly rated response from both poles of an
axis. "One-sided" means only one pole found a good answer to it. The
activation difference between these two sets is the pluralism direction.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config as cfg


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


PRISM_REPO = "HannahRoseKirk/prism-alignment"
PRISM_FILES = {
    "survey": cfg.SURVEY_PATH,
    "conversations": cfg.CONVERSATIONS_PATH,
    "utterances": cfg.UTTERANCES_PATH,
}


def ensure_dataset(force: bool = False) -> bool:
    """Download the three PRISM files if they're not already on disk.

    PRISM is gated on the Hub, so it needs accepting once on the dataset
    page. Returns True once all three files are present.
    """
    cfg.DATASET_DIR.mkdir(parents=True, exist_ok=True)

    missing = {
        name: path
        for name, path in PRISM_FILES.items()
        if force or not path.exists() or path.stat().st_size == 0
    }
    if not missing:
        for name, path in PRISM_FILES.items():
            print(f"  {name:14s} present ({path.stat().st_size / 1e6:.0f} MB)")
        return True

    print(f"missing: {', '.join(missing)}")
    print(f"downloading from {PRISM_REPO}")

    try:
        from huggingface_hub import hf_hub_download

        for name, path in missing.items():
            downloaded = hf_hub_download(
                repo_id=PRISM_REPO, filename=f"{name}.jsonl", repo_type="dataset"
            )
            path.write_bytes(Path(downloaded).read_bytes())
            print(f"  {name:14s} downloaded ({path.stat().st_size / 1e6:.0f} MB)")
        return True
    except Exception as exc:
        print(f"  direct file download failed: {exc}")

    try:
        from datasets import load_dataset

        for name, path in missing.items():
            dataset = load_dataset(PRISM_REPO, name, split="train")
            with open(path, "w", encoding="utf-8") as fh:
                for row in dataset:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {name:14s} written ({path.stat().st_size / 1e6:.0f} MB)")
        return True
    except Exception as exc:
        print(f"  datasets fallback failed: {exc}")

    print("\ncould not fetch PRISM automatically.")
    try:
        from huggingface_hub import list_repo_files

        for f in list_repo_files(PRISM_REPO, repo_type="dataset")[:40]:
            print("   ", f)
    except Exception as exc:
        print(f"could not even list the repo: {exc}")

    print("\nUsually one of:")
    print("  1. PRISM is gated. Accept the terms, then run:")
    print("       from huggingface_hub import notebook_login; notebook_login()")
    print(f"     https://huggingface.co/datasets/{PRISM_REPO}")
    print("  2. No internet in this runtime.")
    print("\nOr copy dataset/ across by hand: survey.jsonl, conversations.jsonl,")
    print("utterances.jsonl into", cfg.DATASET_DIR)

    return all(p.exists() for p in PRISM_FILES.values())


def load_prism():
    """Return (survey, conversations, utterances) as dataframes."""
    if not ensure_dataset():
        raise FileNotFoundError(
            f"PRISM files not available in {cfg.DATASET_DIR}. See the messages above."
        )

    survey = _read_jsonl(cfg.SURVEY_PATH)
    conversations = _read_jsonl(cfg.CONVERSATIONS_PATH)
    utterances = _read_jsonl(cfg.UTTERANCES_PATH)
    print(f"survey       {len(survey):>7,} rows")
    print(f"conversations{len(conversations):>7,} rows")
    print(f"utterances   {len(utterances):>7,} rows")
    return survey, conversations, utterances


def _nested_get(value, key, default=None):
    """A few PRISM fields are nested dicts, occasionally flat instead."""
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def _normalise(text: str) -> str:
    """Casefold and collapse whitespace, for exact prompt matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


# --------------------------------------------------------------------------
# Demographic mapping
# --------------------------------------------------------------------------

def map_demographics(survey: pd.DataFrame) -> dict:
    """Map each user to a pole on each axis.

    Returns {axis: {user_id: pole}}. A user who doesn't fall cleanly on one
    side is left out of that axis rather than guessed at.
    """
    mapping = {axis: {} for axis in cfg.AXES}

    for _, row in survey.iterrows():
        user_id = row.get("user_id")
        if not user_id:
            continue

        region = _nested_get(row.get("location"), "reside_region")
        if region in cfg.WESTERN_REGIONS:
            mapping["origin"][user_id] = "western"
        elif region in cfg.NON_WESTERN_REGIONS:
            mapping["origin"][user_id] = "non_western"

        religion = _nested_get(row.get("religion"), "simplified")
        if religion in cfg.RELIGIOUS_LABELS:
            mapping["religion"][user_id] = "religious"
        elif religion in cfg.SECULAR_LABELS:
            mapping["religion"][user_id] = "secular"

        age = row.get("age")
        if age in cfg.YOUNG_AGE_BRACKETS:
            mapping["age"][user_id] = "young"
        elif age in cfg.OLD_AGE_BRACKETS:
            mapping["age"][user_id] = "old"

    for axis in cfg.AXES:
        pole_a, pole_b = cfg.AXIS_POLES[axis]
        counts = defaultdict(int)
        for pole in mapping[axis].values():
            counts[pole] += 1
        print(
            f"{axis:9s} {pole_a}={counts[pole_a]:,}  {pole_b}={counts[pole_b]:,}  "
            f"(unmapped={len(survey) - len(mapping[axis]):,})"
        )
    return mapping


# --------------------------------------------------------------------------
# Rated responses
# --------------------------------------------------------------------------

def build_rated_responses(conversations: pd.DataFrame, utterances: pd.DataFrame) -> pd.DataFrame:
    """Flatten PRISM into one row per (prompt, model response, rating).

    There's no "role" column in utterances.jsonl. Every row is already one
    model reply to a user_prompt, carrying the score that user gave it. The
    reply text is in model_response, and scores run 1-100.
    """
    user_by_conversation = {}
    for _, row in conversations.iterrows():
        conversation_id = row.get("conversation_id")
        if conversation_id:
            user_by_conversation[conversation_id] = row.get("user_id")

    rows = []
    for _, utterance in utterances.iterrows():
        score = utterance.get("score")
        if score is None or (isinstance(score, float) and np.isnan(score)):
            continue

        conversation_id = utterance.get("conversation_id")
        user_id = utterance.get("user_id") or user_by_conversation.get(conversation_id)
        if not user_id:
            continue

        prompt = str(utterance.get("user_prompt") or "")
        response = str(utterance.get("model_response") or "")
        if not prompt.strip() or not response.strip():
            continue

        rows.append(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "prompt": prompt.strip(),
                "response": response.strip(),
                "score": float(score),
                # Kept for the source-model confound check below: PRISM
                # responses come from 21 different LLMs, and the balanced
                # and one-sided sets need to not be split by model identity.
                "model_name": utterance.get("model_name", ""),
            }
        )

    rated = pd.DataFrame(rows)
    print(f"rated model responses: {len(rated):,}")
    return rated


# --------------------------------------------------------------------------
# Balanced / one-sided labelling
# --------------------------------------------------------------------------

def label_balanced_one_sided(rated: pd.DataFrame, poles: dict, axis: str) -> pd.DataFrame:
    """Tag each liked response as balanced or one-sided, for one axis.

    Every PRISM response is generated fresh inside one user's own
    conversation, so no two users ever see the literal same response text.
    Matching on exact response equality would only ever catch a handful of
    generic refusals and misses everything real, so the match happens on
    the prompt instead: does a highly rated response to this KIND of
    question exist on the other side too, using prompt similarity to decide
    what counts as the same question.

    balanced  - this response's prompt has a match (exact, or cosine >=
                SEMANTIC_MATCH_THRESHOLD) among the other pole's liked
                prompts. Both sides found a good answer to this topic.
    one_sided - no match. Only this pole found a good answer to it.
    """
    pole_a, pole_b = cfg.AXIS_POLES[axis]
    axis_poles = poles[axis]

    liked = rated[rated["score"] >= cfg.HIGH_RATING_THRESHOLD].copy()
    liked["pole"] = liked["user_id"].map(axis_poles)
    liked = liked.dropna(subset=["pole"])

    pool_a = liked[liked["pole"] == pole_a].reset_index(drop=True)
    pool_b = liked[liked["pole"] == pole_b].reset_index(drop=True)

    if not len(pool_a) or not len(pool_b):
        print(f"  {axis}: one side has no liked responses at all, skipping")
        return pd.DataFrame()

    prompts_a = pool_a["prompt"].tolist()
    prompts_b = pool_b["prompt"].tolist()
    normalised_a = [_normalise(p) for p in prompts_a]
    normalised_b = [_normalise(p) for p in prompts_b]

    exact_a = set(normalised_a)
    exact_b = set(normalised_b)

    # Existence check per row (does ANY match on the other side clear the
    # threshold), not a one-to-one pairing. Pool sizes run into the
    # thousands, so a dense len(a) x len(b) similarity matrix could be
    # gigabytes; work through it in chunks and keep the running max.
    vectoriser = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf = vectoriser.fit_transform(prompts_a + prompts_b)
    tfidf_a, tfidf_b = tfidf[: len(prompts_a)], tfidf[len(prompts_a):]

    max_similarity_a_to_b = np.zeros(len(prompts_a))
    max_similarity_b_to_a = np.zeros(len(prompts_b))
    chunk_size = 500
    for start in range(0, len(prompts_a), chunk_size):
        block = cosine_similarity(tfidf_a[start : start + chunk_size], tfidf_b)
        max_similarity_a_to_b[start : start + chunk_size] = block.max(axis=1) if block.size else 0.0
        max_similarity_b_to_a = np.maximum(max_similarity_b_to_a, block.max(axis=0) if block.size else 0.0)

    records = []
    for i, row in pool_a.iterrows():
        matched = (normalised_a[i] in exact_b) or (max_similarity_a_to_b[i] >= cfg.SEMANTIC_MATCH_THRESHOLD)
        records.append(
            {
                "response": row["response"],
                "prompt": row["prompt"],
                "label": "balanced" if matched else "one_sided",
                "liked_by": pole_a,
                "score": row["score"],
                "model_name": row["model_name"],
            }
        )
    for j, row in pool_b.iterrows():
        matched = (normalised_b[j] in exact_a) or (max_similarity_b_to_a[j] >= cfg.SEMANTIC_MATCH_THRESHOLD)
        records.append(
            {
                "response": row["response"],
                "prompt": row["prompt"],
                "label": "balanced" if matched else "one_sided",
                "liked_by": pole_b,
                "score": row["score"],
                "model_name": row["model_name"],
            }
        )

    labelled = pd.DataFrame(records)
    if len(labelled):
        counts = labelled["label"].value_counts().to_dict()
        print(f"  {axis:9s} balanced={counts.get('balanced', 0):,} one_sided={counts.get('one_sided', 0):,}"
              f"  (from {len(pool_a):,} {pole_a} + {len(pool_b):,} {pole_b} liked responses)")
    return labelled


# --------------------------------------------------------------------------
# Pair extraction
# --------------------------------------------------------------------------

def extract_pairs(labelled: pd.DataFrame, axis: str, use_semantic: bool = True) -> pd.DataFrame:
    """Match each balanced response to a one-sided response on a similar prompt.

    Exact prompt text first, then TF-IDF cosine similarity for whatever's
    left. The semantic pass is what makes the Age axis usable at all.
    """
    balanced = labelled[labelled["label"] == "balanced"].reset_index(drop=True)
    one_sided = labelled[labelled["label"] == "one_sided"].reset_index(drop=True)

    if not len(balanced) or not len(one_sided):
        print(f"  {axis}: not enough labelled data to pair")
        return pd.DataFrame()

    pairs = []
    used_one_sided = set()

    # exact prompt matches
    one_sided_by_prompt = defaultdict(list)
    for idx, row in one_sided.iterrows():
        one_sided_by_prompt[_normalise(row["prompt"])].append(idx)

    matched_balanced = set()
    for balanced_idx, balanced_row in balanced.iterrows():
        key = _normalise(balanced_row["prompt"])
        for one_sided_idx in one_sided_by_prompt.get(key, []):
            if one_sided_idx in used_one_sided:
                continue
            pairs.append(
                {
                    "axis": axis,
                    "match_type": "exact",
                    "similarity": 1.0,
                    "prompt": balanced_row["prompt"],
                    "balanced_response": balanced_row["response"],
                    "one_sided_response": one_sided.loc[one_sided_idx, "response"],
                    "one_sided_liked_by": one_sided.loc[one_sided_idx, "liked_by"],
                    "balanced_model": balanced_row.get("model_name", ""),
                    "one_sided_model": one_sided.loc[one_sided_idx, "model_name"],
                }
            )
            used_one_sided.add(one_sided_idx)
            matched_balanced.add(balanced_idx)
            break

    exact_count = len(pairs)

    # semantic matches for whatever's left
    if use_semantic:
        remaining_balanced = [i for i in balanced.index if i not in matched_balanced]
        remaining_one_sided = [i for i in one_sided.index if i not in used_one_sided]

        if remaining_balanced and remaining_one_sided:
            prompts = (
                balanced.loc[remaining_balanced, "prompt"].tolist()
                + one_sided.loc[remaining_one_sided, "prompt"].tolist()
            )
            prompts = [p for p in prompts if p.strip()]
            if len(prompts) >= 2:
                vectoriser = TfidfVectorizer(stop_words="english", min_df=1)
                tfidf = vectoriser.fit_transform(prompts)
                tfidf_balanced = tfidf[: len(remaining_balanced)]
                tfidf_one_sided = tfidf[len(remaining_balanced):]

                similarity = cosine_similarity(tfidf_balanced, tfidf_one_sided)
                for row_i, balanced_idx in enumerate(remaining_balanced):
                    ranked = np.argsort(-similarity[row_i])
                    for col_j in ranked:
                        score = similarity[row_i, col_j]
                        if score < cfg.SEMANTIC_MATCH_THRESHOLD:
                            break
                        one_sided_idx = remaining_one_sided[col_j]
                        if one_sided_idx in used_one_sided:
                            continue
                        pairs.append(
                            {
                                "axis": axis,
                                "match_type": "semantic",
                                "similarity": float(score),
                                "prompt": balanced.loc[balanced_idx, "prompt"],
                                "balanced_response": balanced.loc[balanced_idx, "response"],
                                "one_sided_response": one_sided.loc[one_sided_idx, "response"],
                                "one_sided_liked_by": one_sided.loc[one_sided_idx, "liked_by"],
                                "balanced_model": balanced.loc[balanced_idx, "model_name"],
                                "one_sided_model": one_sided.loc[one_sided_idx, "model_name"],
                            }
                        )
                        used_one_sided.add(one_sided_idx)
                        break

    pairs_df = pd.DataFrame(pairs)
    semantic_count = len(pairs_df) - exact_count
    print(f"  {axis:9s} exact={exact_count:,} semantic={semantic_count:,} total={len(pairs_df):,}")
    return pairs_df


def build_all_pairs(save: bool = True) -> dict:
    """Run the full extraction and return {axis: pairs dataframe}."""
    survey, conversations, utterances = load_prism()
    poles = map_demographics(survey)
    rated = build_rated_responses(conversations, utterances)

    print("\nlabelling balanced vs one-sided")
    pairs_by_axis = {}
    for axis in cfg.AXES:
        labelled = label_balanced_one_sided(rated, poles, axis)
        if not len(labelled):
            continue
        pairs = extract_pairs(labelled, axis)
        if len(pairs) < cfg.MIN_PAIRS_PER_AXIS:
            print(f"  WARNING {axis}: only {len(pairs)} pairs, below minimum {cfg.MIN_PAIRS_PER_AXIS}")
        pairs_by_axis[axis] = pairs
        if save and len(pairs):
            path = cfg.PAIRS_DIR / f"pairs_{axis}.json"
            pairs.to_json(path, orient="records", indent=2, force_ascii=False)
            print(f"  saved {path.name}")
    return pairs_by_axis


def load_pairs(axis: str) -> pd.DataFrame:
    path = cfg.PAIRS_DIR / f"pairs_{axis}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run build_all_pairs() first.")
    return pd.read_json(path)


# --------------------------------------------------------------------------
# Confound diagnostics
# --------------------------------------------------------------------------

def diagnose_pairs(pairs: pd.DataFrame, axis: str) -> dict:
    """Check whether the balanced/one-sided split is confounded.

    If the balanced set differs from the one-sided set in something other
    than pluralism, the vector ends up encoding that other thing instead.
    Two risks worth checking, both raised in the PRISM paper itself:

    Length and formatting. PRISM found these partly explain score
    differences, though weakly (R-squared 0.06), and they capped responses
    at 50 words specifically to limit this. Real, but bounded.

    Source model. Responses come from 21 different LLMs. If the balanced
    set leans toward stronger models and one-sided toward weaker ones, the
    vector separates model quality, not pluralism. Not controlled for by
    the dataset's design, so this one matters more.
    """
    report = {"axis": axis, "n_pairs": len(pairs)}
    if not len(pairs):
        return report

    balanced_length = pairs["balanced_response"].str.split().str.len()
    one_sided_length = pairs["one_sided_response"].str.split().str.len()

    report["balanced_mean_words"] = float(balanced_length.mean())
    report["one_sided_mean_words"] = float(one_sided_length.mean())
    report["word_gap"] = float(balanced_length.mean() - one_sided_length.mean())

    pooled_sd = float(np.sqrt((balanced_length.var() + one_sided_length.var()) / 2))
    report["length_effect_size"] = float(report["word_gap"] / pooled_sd) if pooled_sd else 0.0

    if "balanced_model" in pairs.columns and "one_sided_model" in pairs.columns:
        same_model_rate = (pairs["balanced_model"] == pairs["one_sided_model"]).mean()
        report["same_source_model_rate"] = float(same_model_rate)

    print(f"\n{axis} pair diagnostics")
    print(f"  pairs                {report['n_pairs']:,}")
    print(f"  balanced words       {report['balanced_mean_words']:.1f}")
    print(f"  one-sided words      {report['one_sided_mean_words']:.1f}")
    print(f"  length effect size   {report['length_effect_size']:+.2f}", end="")
    print("   <-- large enough to control for" if abs(report["length_effect_size"]) > 0.2 else "   (small)")
    if "same_source_model_rate" in report:
        print(f"  same source model    {report['same_source_model_rate']:.0%}")
    return report


def diagnose_model_clustering(pairs: pd.DataFrame, axis: str, min_appearances: int = 5) -> pd.DataFrame:
    """Check which source LLMs cluster into balanced vs one-sided.

    A low same_source_model_rate from diagnose_pairs() only matters if
    specific models dominate one side, e.g. a stronger model showing up
    mostly on the balanced side and a weaker one mostly on one-sided - that
    would mean the vector is separating model quality, not pluralism.

    min_appearances filters out models too rare in these ~100-200 pairs to
    say anything meaningful about.
    """
    if not len(pairs) or "balanced_model" not in pairs.columns:
        print(f"{axis}: no model_name data to check")
        return pd.DataFrame()

    balanced_counts = pairs["balanced_model"].value_counts()
    one_sided_counts = pairs["one_sided_model"].value_counts()
    totals = balanced_counts.add(one_sided_counts, fill_value=0).sort_values(ascending=False)

    rows = []
    for model, total in totals.items():
        if total < min_appearances:
            continue
        n_balanced = int(balanced_counts.get(model, 0))
        n_one_sided = int(one_sided_counts.get(model, 0))
        rows.append(
            {
                "model": model,
                "n_total": int(total),
                "n_balanced": n_balanced,
                "n_one_sided": n_one_sided,
                "pct_balanced": round(100 * n_balanced / total, 0),
            }
        )

    result = pd.DataFrame(rows).sort_values("n_total", ascending=False)
    if not len(result):
        print(f"{axis}: no model appears at least {min_appearances} times across "
              f"{len(pairs)} pairs, too sparse to say anything about clustering")
        return result

    # A model at roughly 50/50 isn't contributing to the confound.
    skewed = result[(result["pct_balanced"] >= 75) | (result["pct_balanced"] <= 25)]

    print(f"\n{axis} model clustering (models appearing >= {min_appearances} times)")
    print(result.to_string(index=False))
    if len(skewed):
        print(f"\n  skewed toward one side: {', '.join(skewed['model'].tolist())}")
        print("  these models are pulling the split toward model identity, not just the demographic split")
    else:
        print("\n  no single model is skewed toward one side")
        print("  the low same-source-model rate is spread across many models, not concentrated in a few")

    return result
