"""Activation extraction and the four steering vector estimation methods.

Methods (Im & Li, 2026):
    MoD - Mean of Differences        (Rimsky et al., CAA)
    PoD - PCA of Differences         (Zou et al., RepE)
    PoE - PCA of Embeddings          (Zou et al., RepE)
    CoE - Classifier on Embeddings   (Li et al., ITI)

All four get rescaled to the norm of MoD, so a given alpha pushes by roughly
the same amount no matter which method produced the direction. Without that,
an alpha sweep would be comparing direction times scale, not just direction.
"""

from __future__ import annotations

import gc
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from . import config as cfg


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model(spec, device: str | None = None, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = cfg.MODEL_DTYPE

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding keeps the generated continuation contiguous when batching.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return model, tokenizer, device


def free(model):
    # del model here only drops this function's own reference. Whoever
    # called free() still has their local variable pointing at the same
    # model, so the GPU memory would stay claimed until that variable also
    # goes away. Moving the weights to CPU first releases the GPU memory
    # right now, no matter how long that other reference lives.
    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def n_layers(model) -> int:
    return model.config.num_hidden_layers


def target_layer(model, fraction: float | None = None) -> int:
    """Layer index at the given depth fraction."""
    fraction = cfg.LAYER_DEPTH_FRACTION if fraction is None else fraction
    total = n_layers(model)
    layer = int(round(total * fraction))
    return max(1, min(total, layer))


def layer_module(model, layer_idx: int):
    """Return the decoder block at layer_idx, across common architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        target = model
        try:
            for part in path.split("."):
                target = getattr(target, part)
            return target[layer_idx - 1]
        except (AttributeError, IndexError, TypeError):
            continue
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")


# --------------------------------------------------------------------------
# Activation extraction
# --------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(
    model,
    tokenizer,
    texts,
    layer_idx: int,
    device: str,
    batch_size: int = 8,
    max_length: int = 256,
    pooling: str | None = None,
) -> np.ndarray:
    """Pooled hidden state for each text, at one layer.

    pooling="mean" averages over the real (non-padding) tokens. Reading off
    the very last position of a padded block lands on EOS or padding and
    captures stopping behaviour rather than meaning, so this pools instead.

    pooling="last" takes the final real token, which is what Im & Li use.
    It still resolves the correct position under left or right padding,
    it just skips the averaging.
    """
    pooling = cfg.ACTIVATION_POOLING if pooling is None else pooling

    # Every text must produce exactly one row, in order: the caller pairs
    # row i of the balanced batch with row i of the one-sided batch, so
    # silently dropping an empty string here would shift every pair after it.
    texts = [text if (text and text.strip()) else " " for text in texts]

    pooled_batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        output = model(**encoded, output_hidden_states=True)
        hidden = output.hidden_states[layer_idx]        # (batch, tokens, hidden)
        attention_mask = encoded["attention_mask"]        # (batch, tokens)

        if pooling == "last":
            last_token_idx = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_token_idx, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        pooled_batches.append(pooled.float().cpu().numpy())

    if not pooled_batches:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(pooled_batches, axis=0)


# --------------------------------------------------------------------------
# The four estimators
# --------------------------------------------------------------------------

def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _orient(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """PCA and classifier directions come out with an arbitrary sign, align to MoD."""
    return -candidate if float(np.dot(candidate, reference)) < 0 else candidate


def compute_vectors(balanced_activations: np.ndarray, one_sided_activations: np.ndarray,
                     normalise: bool | None = None) -> dict:
    """The four estimators from Im & Li (2026), eqs 3-6.

        MoD  v = mean(h_balanced - h_one_sided)
        PoD  v = top principal component of (h_balanced - h_one_sided)
        PoE  v = top principal component of h_balanced union h_one_sided
        CoE  v = normal of a linear classifier separating the two sets

    Every vector points from one-sided toward balanced.

    On scale: the paper defines PoD and PoE at unit length and CoE at the
    standard deviation along its own direction, then restores comparability
    by searching a per-method multiplier. With normalise=True (default) all
    three get rescaled to the MoD norm instead, so one alpha is one amount
    of push regardless of method. Set normalise=False for the paper's own
    definitions.
    """
    normalise = cfg.NORMALISE_TO_MOD_NORM if normalise is None else normalise

    supplied_count = min(len(balanced_activations), len(one_sided_activations))
    if supplied_count < 2:
        raise ValueError(
            f"need at least 2 pairs per side, got balanced={len(balanced_activations)} "
            f"one_sided={len(one_sided_activations)}"
        )
    balanced_activations = balanced_activations[:supplied_count]
    one_sided_activations = one_sided_activations[:supplied_count]

    # Drop any pair where either side is non-finite, keeping the two arrays
    # aligned. A few small models produce activation spikes that overflow
    # into inf, and sklearn raises on that. Filtering must happen jointly:
    # dropping a bad row from only one array would shift every later pair
    # out of alignment.
    finite = np.isfinite(balanced_activations).all(axis=1) & np.isfinite(one_sided_activations).all(axis=1)
    if not finite.all():
        dropped = int((~finite).sum())
        print(f"    dropping {dropped} of {supplied_count} pairs with non-finite activations")
        balanced_activations = balanced_activations[finite]
        one_sided_activations = one_sided_activations[finite]
        if len(balanced_activations) < 2:
            raise ValueError(
                f"only {len(balanced_activations)} finite pairs left after dropping {dropped}; "
                "this model is producing overflow at this layer"
            )

    # Everything below counts rows that survived the filter, not the rows
    # that came in. The CoE label vector has to match the pooled activation
    # matrix row for row, so reusing the pre-filter count here would make
    # LogisticRegression.fit() raise on a length mismatch.
    pair_count = len(balanced_activations)

    # MoD (eq 3). Theorem 3.1: this is the vector that minimises the
    # steering objective, so every other method is measured against it.
    differences = balanced_activations - one_sided_activations
    mod_vector = differences.mean(axis=0)
    mod_norm = float(np.linalg.norm(mod_vector))
    if mod_norm == 0:
        raise ValueError("MoD vector is zero; balanced and one-sided activations are identical")

    vectors_by_method = {"MoD": mod_vector}
    pooled_activations = np.concatenate([balanced_activations, one_sided_activations], axis=0)

    def scale_to(direction, paper_norm):
        unit_direction = _unit(_orient(direction, mod_vector))
        return unit_direction * (mod_norm if normalise else paper_norm)

    # PoD (eq 4): top PC of the difference vectors, unit length in the paper.
    pod_direction = PCA(n_components=1).fit(differences).components_[0]
    vectors_by_method["PoD"] = scale_to(pod_direction, 1.0)

    # PoE (eq 5): top PC of the pooled activations, unit length in the paper.
    poe_direction = PCA(n_components=1).fit(pooled_activations).components_[0]
    vectors_by_method["PoE"] = scale_to(poe_direction, 1.0)

    # CoE (eq 6): classifier normal, scaled to the standard deviation of the
    # activations projected along it, following ITI.
    labels = np.concatenate([np.ones(pair_count), np.zeros(pair_count)])
    classifier = LogisticRegression(max_iter=2000, C=1.0).fit(pooled_activations, labels)
    coe_direction = classifier.coef_[0]
    coe_projected_std = float(np.std(pooled_activations @ _unit(coe_direction)))
    vectors_by_method["CoE"] = scale_to(coe_direction, coe_projected_std)
    vectors_by_method["_coe_train_accuracy"] = float(classifier.score(pooled_activations, labels))
    vectors_by_method["_n_pairs_used"] = pair_count

    return vectors_by_method


def random_control(reference: np.ndarray, seed: int = cfg.SEED) -> np.ndarray:
    """A random direction at the same norm as reference, for the control test.

    If steering with this moves the output as much as the real vector does,
    the effect is coming from the perturbation, not the demographic signal.
    """
    rng = np.random.default_rng(seed)
    random_direction = rng.normal(size=reference.shape)
    return _unit(random_direction) * float(np.linalg.norm(reference))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_vectors_for_model(spec, pairs_by_axis: dict, layer_fraction=None, save=True) -> dict:
    """Extract activations and compute all four vectors, for every axis."""
    model, tokenizer, device = load_model(spec)

    # Same reason as stage_generate: an interrupted cell must not leave the
    # weights sitting on the GPU for the next model to collide with.
    try:
        layer = target_layer(model, layer_fraction)
        total_layers = n_layers(model)
        print(f"{spec.short_name}: {total_layers} layers, injecting at layer {layer} ({layer/total_layers:.0%})")

        result = {"model": spec.short_name, "layer": layer, "n_layers": total_layers, "axes": {}}

        for axis, pairs in pairs_by_axis.items():
            if pairs is None or not len(pairs):
                continue

            usable_pairs = pairs.dropna(subset=["balanced_response", "one_sided_response"])
            usable_pairs = usable_pairs[
                (usable_pairs["balanced_response"].str.strip() != "")
                & (usable_pairs["one_sided_response"].str.strip() != "")
            ]
            if not len(usable_pairs):
                print(f"  {axis}: no usable pairs after dropping blanks")
                continue

            balanced_responses = usable_pairs["balanced_response"].tolist()
            one_sided_responses = usable_pairs["one_sided_response"].tolist()

            balanced_activations = extract_activations(model, tokenizer, balanced_responses, layer, device)
            one_sided_activations = extract_activations(model, tokenizer, one_sided_responses, layer, device)

            try:
                vectors_by_method = compute_vectors(balanced_activations, one_sided_activations)
            except ValueError as exc:
                print(f"  {axis}: skipped ({exc})")
                continue

            coe_accuracy = vectors_by_method.pop("_coe_train_accuracy", None)
            pair_count = vectors_by_method.pop("_n_pairs_used")
            vectors_by_method["Random"] = random_control(vectors_by_method["MoD"])

            # Typical size of the state the vector gets added to. A vector norm
            # on its own says nothing about how hard the model is being pushed:
            # 128 is a nudge against a hidden state of 500 and an overwrite
            # against one of 150. Recording it here lets the steering code hold
            # the ratio constant instead of the absolute length, which is the
            # only way one setting means the same thing across axes and models.
            hidden_norm = float(np.median(np.linalg.norm(balanced_activations, axis=1)))

            result["axes"][axis] = {
                "n_pairs": int(pair_count),
                "coe_train_accuracy": coe_accuracy,
                "hidden_norm": hidden_norm,
                "vectors": {name: vector.tolist() for name, vector in vectors_by_method.items()},
                "norms": {name: float(np.linalg.norm(vector)) for name, vector in vectors_by_method.items()},
            }
            mod_norm = float(np.linalg.norm(vectors_by_method["MoD"]))
            print(f"  {axis:9s} pairs={pair_count:,} |MoD|={mod_norm:.3f} "
                  f"|h|={hidden_norm:.1f} push@1.5={150 * mod_norm / hidden_norm:.0f}% "
                  f"coe_accuracy={coe_accuracy:.2f}")
    finally:
        free(model)

    if save:
        path = vectors_path(spec, layer_fraction)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
        print(f"  saved {path.name}")
    return result


def vectors_path(spec, layer_fraction: float | None = None):
    """Cache file for one model's vectors, keyed by model AND layer depth.

    Keying only by model name would let a layer sweep silently reuse
    whatever layer the vectors were first built at, since a cache hit for
    "this model" would look valid regardless of which depth was asked for.
    """
    resolved_fraction = cfg.LAYER_DEPTH_FRACTION if layer_fraction is None else layer_fraction
    depth_tag = f"L{round(resolved_fraction * 100):02d}"
    return cfg.VECTORS_DIR / f"vectors_{spec.slug}_{depth_tag}.json"


def load_vectors(spec, layer_fraction: float | None = None) -> dict:
    path = vectors_path(spec, layer_fraction)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run build_vectors_for_model() first.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_vector(bundle: dict, axis: str, method: str) -> np.ndarray:
    return np.asarray(bundle["axes"][axis]["vectors"][method], dtype=np.float32)
