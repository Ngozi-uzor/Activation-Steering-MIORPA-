"""End-to-end orchestration with checkpointing.

Stages:
    stage_pairs()      PRISM -> contrastive pairs
    stage_benchmarks() build leakage-filtered eval question sets
    stage_vectors()    activations -> MoD / PoD / PoE / CoE per model
    stage_generate()   steered + baseline generations
    stage_evaluate()   bertscore + perplexity
    run_all()          all of the above

Generation checkpoints after every condition, so a Colab disconnect costs
one condition rather than the whole run. Re-running skips finished work.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import benchmarks, data, evaluate, steering, vectors


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def _checkpoint_path(tag: str, model_slug: str) -> Path:
    return cfg.GENERATIONS_DIR / f"gen_{tag}_{model_slug}.csv"


def _done_conditions(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path)
    except Exception:
        return set()
    if not len(existing) or "condition_key" not in existing.columns:
        return set()
    return set(existing["condition_key"].unique())


def _append_rows(path: Path, rows: list):
    new_rows = pd.DataFrame(rows)
    write_header = not path.exists()
    new_rows.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8")


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_pairs(force: bool = False) -> dict:
    print("=" * 70)
    print("STAGE 1  contrastive calibration pairs")
    print("=" * 70)

    cached_pairs = {}
    if not force:
        for axis in cfg.AXES:
            try:
                cached_pairs[axis] = data.load_pairs(axis)
            except FileNotFoundError:
                cached_pairs = {}
                break
    if cached_pairs:
        for axis, pairs in cached_pairs.items():
            print(f"  {axis:9s} {len(pairs):,} pairs (cached)")
        return cached_pairs

    return data.build_all_pairs(save=True)


def stage_benchmarks(per_axis: int | None = None, force: bool = False) -> dict:
    print("=" * 70)
    print("STAGE 2  zero-shot evaluation sets")
    print("=" * 70)

    if not force:
        try:
            eval_sets = {axis: benchmarks.load_eval_set(axis) for axis in cfg.AXES}
            for axis, questions in eval_sets.items():
                print(f"  {axis:9s} {len(questions):,} questions (cached)")
            return eval_sets
        except FileNotFoundError:
            pass

    return benchmarks.build_eval_sets(per_axis=per_axis, save=True)


def stage_vectors(pairs_by_axis: dict, model_names=None, layer_fraction=None, force=False) -> dict:
    print("=" * 70)
    print("STAGE 3  steering vectors")
    print("=" * 70)

    model_names = model_names or [m.short_name for m in cfg.MODELS]
    vector_bundles = {}
    for name in model_names:
        spec = cfg.MODELS_BY_NAME[name]
        if not force:
            try:
                vector_bundles[name] = vectors.load_vectors(spec, layer_fraction=layer_fraction)
                print(f"  {name} (cached, layer_fraction={cfg.LAYER_DEPTH_FRACTION if layer_fraction is None else layer_fraction})")
                continue
            except FileNotFoundError:
                pass
        vector_bundles[name] = vectors.build_vectors_for_model(
            spec, pairs_by_axis, layer_fraction=layer_fraction, save=True
        )
    return vector_bundles


def stage_generate(run_cfg: cfg.RunConfig, eval_sets: dict, vector_bundles: dict) -> dict:
    """Generate baseline and steered answers for every condition."""
    print("=" * 70)
    print("STAGE 4  generation")
    print("=" * 70)
    print(run_cfg.describe())

    checkpoint_paths = {}
    for name in run_cfg.models:
        spec = cfg.MODELS_BY_NAME[name]
        path = _checkpoint_path(run_cfg.tag, spec.slug)
        checkpoint_paths[name] = path

        bundle = vector_bundles.get(name)
        if bundle is None:
            print(f"  {name}: no vectors, skipping")
            continue

        methods = list(run_cfg.methods) + (["Random"] if run_cfg.include_random_control else [])
        wanted_conditions = []
        for axis in run_cfg.axes:
            if axis not in bundle["axes"]:
                continue
            questions = eval_sets[axis][: run_cfg.questions_per_axis]
            wanted_conditions.append((axis, "baseline", 0.0, questions))
            for method in methods:
                for alpha in run_cfg.alphas:
                    if alpha == 0.0:
                        continue
                    wanted_conditions.append((axis, method, alpha, questions))

        already_done = _done_conditions(path)
        remaining = [c for c in wanted_conditions if f"{c[0]}|{c[1]}|{c[2]}" not in already_done]
        if not remaining:
            print(f"  {name}: all {len(wanted_conditions)} conditions already done")
            continue

        print(f"\n  {name}: {len(remaining)} of {len(wanted_conditions)} conditions to run")
        model, tokenizer, device = vectors.load_model(spec)

        # free() has to run even if this model's loop dies part way through.
        # Interrupting a Colab cell is the common case, and without this the
        # weights stay on the GPU, so the next model loads into whatever is
        # left and hits an out-of-memory error that looks unrelated.
        try:
            layer = vectors.target_layer(model, run_cfg.layer_fraction)
            print(f"  layer {layer}/{vectors.n_layers(model)}  device {device}")

            for axis, method, alpha, questions in remaining:
                condition_key = f"{axis}|{method}|{alpha}"
                started_at = time.time()

                vector = None if method == "baseline" else vectors.get_vector(bundle, axis, method)

                # A non-finite vector turns into NaN logits during generation,
                # which surfaces as a CUDA device-side assert. That error tends
                # to poison the CUDA context for the rest of the process, so
                # every condition after it fails too, not just the bad one.
                # Catching it here means one bad vector costs one skipped
                # condition instead of the whole run.
                if vector is not None and not np.isfinite(vector).all():
                    print(f"    {condition_key:28s} SKIPPED: vector contains non-finite values")
                    continue

                # What reaches the model is alpha * |v|, but what matters is
                # that measured against |h|, the state it is added to. Holding
                # the ratio constant is the only way one setting means the same
                # thing on an axis whose vectors are long and one whose vectors
                # are short, or on models whose representations differ in scale
                # by more than twenty times.
                vector_norm = float(np.linalg.norm(vector)) if vector is not None else 0.0
                applied_alpha = alpha
                if run_cfg.equalise_push and vector_norm > 0:
                    hidden_norm = bundle["axes"][axis].get("hidden_norm")
                    if not hidden_norm:
                        raise RuntimeError(
                            f"{name}/{axis}: this vector file predates hidden_norm, which "
                            "equalise_push needs. Delete results/vectors and re-run "
                            "stage_vectors to rebuild them."
                        )
                    # alpha scales the target so a sweep still works; at
                    # DEFAULT_ALPHA the push is exactly PUSH_TARGET_RATIO.
                    target = cfg.PUSH_TARGET_RATIO * hidden_norm * (alpha / cfg.DEFAULT_ALPHA)
                    applied_alpha = target / vector_norm

                try:
                    responses = steering.generate(
                        model, tokenizer, questions, device,
                        layer_idx=layer, vector=vector, alpha=applied_alpha,
                    )
                except Exception as exc:
                    print(f"    {condition_key:28s} FAILED: {exc}")
                    if "device-side assert" in str(exc) or "CUDA error" in str(exc):
                        print("    CUDA is likely poisoned for the rest of this session.")
                        print("    Restart the Colab runtime, then re-run - checkpointing")
                        print("    will skip everything already saved.")
                        raise
                    continue

                # The two diagnostic columns are only written when the push is
                # being equalised. Adding them unconditionally would change
                # the CSV header, and _append_rows appends to whatever file is
                # already there, so an existing checkpoint would end up with
                # its columns misaligned halfway down.
                extra = {}
                if run_cfg.equalise_push:
                    extra = {"vector_norm": vector_norm, "applied_alpha": applied_alpha,
                             "hidden_norm": bundle["axes"][axis].get("hidden_norm"),
                             "push_ratio": (applied_alpha * vector_norm
                                            / bundle["axes"][axis]["hidden_norm"]
                                            if vector is not None else 0.0)}

                _append_rows(path, [
                    {
                        "model": name,
                        "axis": axis,
                        "method": method,
                        "alpha": alpha,
                        "layer": layer,
                        "condition": "baseline" if method == "baseline" else f"{method}@{alpha}",
                        "condition_key": condition_key,
                        "question": question,
                        "response": response,
                        **extra,
                    }
                    for question, response in zip(questions, responses)
                ])

                push_note = ""
                if run_cfg.equalise_push and vector is not None:
                    ratio = applied_alpha * vector_norm / bundle["axes"][axis]["hidden_norm"]
                    push_note = f"  |v|={vector_norm:.1f} alpha->{applied_alpha:.3f} push={ratio:.0%}"
                print(f"    {condition_key:28s} {len(questions):>4} gens  "
                      f"{time.time()-started_at:>6.1f}s{push_note}")
        finally:
            vectors.free(model)

        print(f"  saved {path.name}")

    return checkpoint_paths


def stage_evaluate(run_cfg: cfg.RunConfig):
    """Score every generation file with BERTScore and perplexity, write the summary.

    Pluralism isn't scored here, that runs afterwards through judge.py on
    this stage's output.
    """
    print("=" * 70)
    print("STAGE 5  evaluation")
    print("=" * 70)

    generation_frames = []
    for name in run_cfg.models:
        spec = cfg.MODELS_BY_NAME[name]
        path = _checkpoint_path(run_cfg.tag, spec.slug)
        if path.exists():
            generation_frames.append(pd.read_csv(path))
        else:
            print(f"  {name}: no generations found")

    if not generation_frames:
        raise RuntimeError("nothing to evaluate; run stage_generate first")

    generations = pd.concat(generation_frames, ignore_index=True)
    generations["response"] = generations["response"].fillna("").astype(str)
    print(f"scoring {len(generations):,} generations")

    scored = evaluate.score_dataframe(generations)

    scored_path = cfg.EVAL_DIR / f"scored_{run_cfg.tag}.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8")
    print(f"saved {scored_path.name}")

    summary = evaluate.summarise(scored)
    summary_path = cfg.EVAL_DIR / f"summary_{run_cfg.tag}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"saved {summary_path.name}")

    return scored, summary


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------

def run_all(run_cfg: cfg.RunConfig | None = None):
    run_cfg = run_cfg or cfg.RunConfig()
    started_at = time.time()

    pairs = stage_pairs()
    eval_sets = stage_benchmarks(per_axis=run_cfg.questions_per_axis)
    vector_bundles = stage_vectors(pairs, model_names=run_cfg.models, layer_fraction=run_cfg.layer_fraction)
    stage_generate(run_cfg, eval_sets, vector_bundles)
    scored, summary = stage_evaluate(run_cfg)

    print("=" * 70)
    print(f"done in {(time.time()-started_at)/60:.1f} min")
    print("=" * 70)
    return scored, summary


# --------------------------------------------------------------------------
# Ablations
# --------------------------------------------------------------------------

def ablation_alpha_sweep(model_name: str):
    """How much steering the language survives before it breaks.

    Perplexity and BERTScore against alpha give the upper bound: where
    perplexity starts climbing and BERTScore starts falling is where the
    model is degrading. Picking the best alpha inside that range is a
    pluralism question, which needs judge.py.
    """
    run_cfg = cfg.RunConfig(
        models=[model_name],
        methods=("MoD",),
        alphas=cfg.ALPHA_SWEEP,
        questions_per_axis=40,
        include_random_control=False,
        tag=f"alpha_sweep_{cfg.MODELS_BY_NAME[model_name].slug}",
    )
    return run_all(run_cfg)


def ablation_layer_sweep(model_name: str):
    """Where in the network the cultural signal actually lives."""
    summaries = []
    for layer_fraction in cfg.LAYER_SWEEP_FRACTIONS:
        print(f"\n### layer fraction {layer_fraction:.0%}")
        run_cfg = cfg.RunConfig(
            models=[model_name],
            methods=("MoD",),
            alphas=(cfg.DEFAULT_ALPHA,),
            layer_fraction=layer_fraction,
            questions_per_axis=40,
            include_random_control=False,
            tag=f"layer_{int(layer_fraction*100)}_{cfg.MODELS_BY_NAME[model_name].slug}",
        )
        _, summary = run_all(run_cfg)
        summary["layer_fraction"] = layer_fraction
        summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    path = cfg.EVAL_DIR / f"layer_sweep_{cfg.MODELS_BY_NAME[model_name].slug}.csv"
    combined.to_csv(path, index=False, encoding="utf-8")
    print(f"saved {path.name}")
    return combined


def ablation_equalised_push(model_name: str, judge_it: bool = True):
    """Shared alpha versus equal push, on the same model and questions.

    The naive setup gives every axis the same alpha, so every axis gets a
    different actual push because the vectors have different lengths. This runs
    that setup and the corrected one back to back, which makes the difference
    between them the evidence that the axis gap was a scaling artefact.

    BERTScore alone cannot finish the argument. It says whether the answer is
    still about the question, so it can only show that a collapsed axis stopped
    collapsing. Whether the recovered answers are actually more pluralistic is
    a judge question, so with judge_it the same two judges that scored the main
    run score both conditions here.
    """
    from . import judge

    summaries, resolved = [], []
    for equalise in (False, True):
        label = "equal push" if equalise else "shared alpha"
        print(f"\n### {label}")
        tag = f"push_{'equal' if equalise else 'shared'}_{cfg.MODELS_BY_NAME[model_name].slug}"
        run_cfg = cfg.RunConfig(
            models=[model_name],
            methods=("MoD",),
            alphas=(cfg.DEFAULT_ALPHA,),
            # Matches the main run. At 60 the tie rates left too few decided
            # pairs for the win rates to mean anything.
            questions_per_axis=cfg.EVAL_QUESTIONS_PER_AXIS,
            include_random_control=False,
            equalise_push=equalise,
            tag=tag,
        )
        scored, summary = run_all(run_cfg)
        summary["push_mode"] = label
        summaries.append(summary)

        if not judge_it:
            continue

        pairs = judge.build_pairwise_set(scored, tag=tag)
        for judge_key in judge.DEFAULT_JUDGES:
            judged = judge.run_judge(pairs, judge_key, tag=tag)
            verdicts = judge.resolve_pairwise(judged, tag=tag)
            verdicts["push_mode"] = label
            resolved.append(verdicts)
            judge.unload_judge()

    combined = pd.concat(summaries, ignore_index=True)
    path = cfg.EVAL_DIR / f"push_comparison_{cfg.MODELS_BY_NAME[model_name].slug}.csv"
    combined.to_csv(path, index=False, encoding="utf-8")
    print(f"saved {path.name}")

    steered = combined[combined["method"] == "MoD"]
    print("\ncoherence: shared alpha vs equal push, per axis")
    print(steered.pivot_table(index="axis", columns="push_mode",
                              values=["bertscore", "perplexity"]).round(3).to_string())

    if resolved:
        verdicts = pd.concat(resolved, ignore_index=True)
        path = cfg.EVAL_DIR / f"push_judged_{cfg.MODELS_BY_NAME[model_name].slug}.csv"
        verdicts.to_csv(path, index=False, encoding="utf-8")
        print(f"saved {path.name}")

        print("\npluralism: steered win rate against baseline, per axis")
        for judge_key, block in verdicts.groupby("judge"):
            rates = block.groupby(["axis", "push_mode"])["verdict"].apply(
                lambda v: round((v == "steered").sum()
                                / max((v != "tie").sum(), 1), 3)
            )
            print(f"\n  {judge_key}")
            print(rates.unstack().to_string())
        print("\nAnything above 0.50 means the judge preferred the steered answer.")
        print("Religion rising from near zero under equal push is the result")
        print("this ablation exists to test.")

    return combined


def measure_push_ratio(model_name: str, pairs_by_axis: dict, layer_fraction=None):
    """How big the steering push is next to the hidden state it is added to.

    |alpha * v| on its own says nothing without a scale to compare it to.
    This measures the typical |h| at the injection layer and reports the push
    as a fraction of it, which is the number that says whether the model was
    nudged or drowned.
    """
    spec = cfg.MODELS_BY_NAME[model_name]
    bundle = vectors.load_vectors(spec, layer_fraction=layer_fraction)
    model, tokenizer, device = vectors.load_model(spec)

    try:
        layer = bundle["layer"]
        rows = []
        for axis, axis_data in bundle["axes"].items():
            pairs = pairs_by_axis.get(axis)
            if pairs is None or not len(pairs):
                continue

            texts = pairs["balanced_response"].dropna().tolist()[:60]
            activations = vectors.extract_activations(model, tokenizer, texts, layer, device)
            hidden_norm = float(np.median(np.linalg.norm(activations, axis=1)))

            vector_norm = float(np.linalg.norm(vectors.get_vector(bundle, axis, "MoD")))
            push = cfg.DEFAULT_ALPHA * vector_norm
            rows.append({
                "axis": axis,
                "vector_norm": round(vector_norm, 2),
                "hidden_norm": round(hidden_norm, 2),
                "push": round(push, 2),
                "push_as_pct_of_hidden": round(100 * push / hidden_norm, 1) if hidden_norm else np.nan,
            })
    finally:
        vectors.free(model)

    report = pd.DataFrame(rows)
    path = cfg.EVAL_DIR / f"push_ratio_{spec.slug}.csv"
    report.to_csv(path, index=False, encoding="utf-8")
    print(f"\npush size relative to the hidden state, alpha={cfg.DEFAULT_ALPHA}")
    print(report.to_string(index=False))
    print(f"saved {path.name}")
    return report


def ablation_random_control(model_name: str):
    """The control that decides whether any of this is real.

    A random vector of the same norm shouldn't move the output the way the
    real vector does. On the automatic metrics alone this only shows
    whether the real vector perturbs the model differently from noise.
    Whether it perturbs it toward pluralism specifically needs the judge.
    """
    run_cfg = cfg.RunConfig(
        models=[model_name],
        methods=("MoD",),
        alphas=(cfg.DEFAULT_ALPHA,),
        questions_per_axis=60,
        include_random_control=True,
        tag=f"control_{cfg.MODELS_BY_NAME[model_name].slug}",
    )
    scored, summary = run_all(run_cfg)

    print("\nreal vector vs random control")
    comparison = summary[summary["method"].isin(["MoD", "Random", "baseline"])]
    print(comparison[["axis", "method", "bertscore", "perplexity", "n_words"]].to_string(index=False))
    return scored, summary
