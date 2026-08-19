"""Settings for the MIORPA pluralistic steering pipeline.

Everything tunable lives here so the notebook stays thin and every stage
reads the same values.
"""

from dataclasses import dataclass, field
from pathlib import Path


def _default_model_dtype():
    """fp32, not fp16.

    fp16 tops out around 65,504. Small models sometimes produce activation
    spikes in a few hidden dimensions that go past that, which turns into
    inf. sklearn's PCA and logistic regression crash on inf, and NaN logits
    during steered generation trigger a CUDA device-side assert. fp32 has
    enough headroom that this doesn't happen, and it costs almost nothing
    extra for models this size on a normal GPU.
    """
    import torch

    return torch.float32


MODEL_DTYPE = _default_model_dtype()


def _enable_tf32():
    """Let matmuls on Ampere+ GPUs use TF32 instead of full fp32.

    TF32 keeps fp32's exponent range, so none of the overflow behaviour
    _default_model_dtype() above is guarding against comes back. It only
    trims mantissa bits, which cuts matmul time substantially on Ampere and
    newer cards and costs nothing on GPUs that don't support it.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


_enable_tf32()


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
SURVEY_PATH = DATASET_DIR / "survey.jsonl"
CONVERSATIONS_PATH = DATASET_DIR / "conversations.jsonl"
UTTERANCES_PATH = DATASET_DIR / "utterances.jsonl"

RESULTS_DIR = ROOT / "results"
PAIRS_DIR = RESULTS_DIR / "pairs"
VECTORS_DIR = RESULTS_DIR / "vectors"
GENERATIONS_DIR = RESULTS_DIR / "generations"
EVAL_DIR = RESULTS_DIR / "evaluation"
BENCHMARK_DIR = RESULTS_DIR / "benchmarks"

for output_dir in (RESULTS_DIR, PAIRS_DIR, VECTORS_DIR, GENERATIONS_DIR, EVAL_DIR, BENCHMARK_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Demographic axes
# --------------------------------------------------------------------------
# Origin   -> Western vs Non-Western
# Religion -> Religious vs Secular
# Age      -> Young (Gen Y/Z) vs Old (Gen X/Boomers)

WESTERN_REGIONS = {"Europe", "Americas", "Oceania"}
NON_WESTERN_REGIONS = {"Africa", "Asia"}

# Counts in dataset/survey.jsonl, religion.simplified:
# No Affiliation 851, Christian 487, Prefer not to say 59, Jewish 42,
# Muslim 31, Other 30. "Prefer not to say" sits in neither set on purpose,
# those users are dropped from this axis rather than guessed at.
RELIGIOUS_LABELS = {"Christian", "Muslim", "Jewish", "Hindu", "Buddhist", "Other"}
SECULAR_LABELS = {"No Affiliation"}

YOUNG_AGE_BRACKETS = {"18-24 years old", "25-34 years old", "35-44 years old"}
OLD_AGE_BRACKETS = {"45-54 years old", "55-64 years old", "65+ years old"}

AXES = ("origin", "religion", "age")

AXIS_POLES = {
    "origin": ("western", "non_western"),
    "religion": ("religious", "secular"),
    "age": ("young", "old"),
}


# --------------------------------------------------------------------------
# Calibration pairs
# --------------------------------------------------------------------------

# Cosine threshold for matching questions by meaning once exact text match
# runs out. Age only has 13 exact-match pairs, too thin on its own.
SEMANTIC_MATCH_THRESHOLD = 0.75

# A response counts as liked at or above this score. PRISM scores run 1-100.
HIGH_RATING_THRESHOLD = 75

MIN_PAIRS_PER_AXIS = 30


# --------------------------------------------------------------------------
# Models under test
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    short_name: str
    family: str
    params: str

    @property
    def slug(self) -> str:
        return self.short_name.replace("/", "_").replace(".", "-")


MODELS = [
    ModelSpec("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M", "smollm2", "135M"),
    ModelSpec("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2-360M", "smollm2", "360M"),
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2-1.7B", "smollm2", "1.7B"),
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B", "qwen2.5", "0.5B"),
    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "qwen2.5", "1.5B"),
    ModelSpec("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B", "qwen2.5", "3B"),
]

MODELS_BY_NAME = {m.short_name: m for m in MODELS}


# --------------------------------------------------------------------------
# Steering
# --------------------------------------------------------------------------

VECTOR_METHODS = ("MoD", "PoD", "PoE", "CoE")

# Injection depth as a fraction of total layers, so it carries over to models
# with different depths. Im & Li (2026) steer Llama-2-7b-chat at layer 13 of
# 32, which is about 0.41. The layer sweep checks if that holds here too.
LAYER_DEPTH_FRACTION = 0.45

# Multiplier set matches Im & Li: 0.5 to 3.0, plus 0 as the unsteered point.
DEFAULT_ALPHA = 1.5
ALPHA_SWEEP = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
LAYER_SWEEP_FRACTIONS = (0.25, 0.41, 0.55, 0.70, 0.85)

# How activations get pooled before building a vector.
#   "mean" - average over every real token. Stable, default.
#   "last" - final token only, which is what Im & Li use.
ACTIVATION_POOLING = "mean"

# Rescale PoD, PoE and CoE to the same norm as MoD before steering.
#
# Im & Li define PoD and PoE at unit length and CoE at the standard
# deviation along its own direction, then search a separate multiplier per
# method to make them comparable. Rescaling to |MoD| does the same job
# directly, so one alpha means the same push for every method.
#
# Set False to use the paper's own scaling instead. Their result that CoE
# underperforms is partly because its natural magnitude is small, so this
# flag changes how the four methods rank against each other.
NORMALISE_TO_MOD_NORM = True

# Equalising the push across axes.
#
# alpha only multiplies the vector; it does not set the strength on its own.
# Each axis's vector carries whatever length its own activations produced -
# on Qwen2.5-1.5B that is |MoD| = 14 for origin, 40 for age, 85 for religion.
# One shared alpha therefore pushes religion about six times harder than
# origin, so comparing axes at "the same alpha" compares three different
# interventions, not three axes.
#
# With RunConfig.equalise_push set, alpha is rescaled per axis so that
# alpha * |v| equals alpha * PUSH_REFERENCE_NORM everywhere. The reference is
# origin's norm because that is the axis that stayed coherent, so it is the
# strength already known not to break the model.
PUSH_REFERENCE_NORM = 14.0

# Steer only the tokens being generated, never the prompt itself.
STEER_PROMPT_TOKENS = False

MAX_NEW_TOKENS = 128


def _default_generation_batch_size() -> int:
    """Pick a batch size from whatever GPU is actually attached.

    Colab hands out a different card depending on availability, so this
    reads the real memory instead of assuming one card every time.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 16
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_gb >= 70:      # A100 80GB, H100
            return 96
        if total_gb >= 35:      # A100 40GB
            return 64
        if total_gb >= 20:      # V100 32GB
            return 32
        return 16                # T4 16GB or unknown
    except Exception:
        return 16


GENERATION_BATCH_SIZE = _default_generation_batch_size()
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 20260802


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
# Two fully-automatic metrics:
#   bertscore   - did the answer stay on topic
#   perplexity  - is the answer still fluent English
#
# Pluralism is scored separately, by the pairwise judges in judge.py, on
# this stage's output.

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
BERTSCORE_FALLBACK = "roberta-large"

# One fixed scorer for every model, otherwise a 3B always looks more fluent
# than a 135M by construction and the comparison means nothing.
PERPLEXITY_SCORER = "gpt2-large"

# Drop any test question too close to a calibration question.
LEAKAGE_MODEL = "all-MiniLM-L6-v2"
LEAKAGE_THRESHOLD = 0.50

EVAL_QUESTIONS_PER_AXIS = 200
CHECKPOINT_EVERY = 100


# --------------------------------------------------------------------------
# Pairwise judging
# --------------------------------------------------------------------------
# Judging is pairwise (baseline vs steered on the same question), not an
# absolute 1-10 score. Absolute scores from LLM judges bunch around 7 and 8
# and the fine distinctions are mostly noise; a forced comparison between
# two answers to the same question is a far more reliable instrument and
# answers the research question directly.
#
# Neither judge comes from a family under test. LLM judges rate their own
# family's outputs higher, and half the test set is Qwen, so a Qwen judge
# would quietly advantage three of the six models and corrupt the scaling
# comparison. Same reason rules out anything from HuggingFaceTB.

# How many pairs per condition go to the judge. Every pair is judged twice,
# once in each order, so the real call count is double this.
#
# Set to the full question set rather than a sample. At 60 the win rates were
# resting on very few pairs once ties were removed - Qwen2.5-3B origin gave a
# decisive rate of 0.667 from 6 wins and 3 losses, with the other 51 pairs
# tied and discarded. Nine decisions cannot separate a real effect from
# chance. Judging everything roughly triples the decided pairs and narrows
# every interval by about a factor of 1.8, at the cost of a longer judging
# stage. Confidence intervals are still reported, because high tie rates mean
# even the full set leaves some conditions thin.
JUDGE_PAIRS_PER_CONDITION = EVAL_QUESTIONS_PER_AXIS

def _default_judge_batch_size() -> int:
    """Pick a judge batch size from the attached GPU, same idea as generation.

    Judge prompts carry a full question plus two responses each, and
    Mistral-Small is 24B, so headroom per example is smaller than for the
    small models under test - scaled down accordingly at every tier.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 4
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_gb >= 70:      # A100 80GB, H100
            return 32
        if total_gb >= 35:      # A100 40GB
            return 16
        if total_gb >= 20:      # V100 32GB
            return 8
        return 4                 # T4 16GB or unknown
    except Exception:
        return 4


# Fallback only. judge.run_judge() normally sizes the batch from the memory
# actually free once a judge's weights are resident, since a 7B and a 24B
# judge have very different headroom on the same card.
JUDGE_BATCH_SIZE = _default_judge_batch_size()

# The chat judges answer with one line of JSON, about 40 tokens. A batch keeps
# stepping until every sequence in it has finished, so a single rambling reply
# drags the whole batch to the cap. 128 still leaves roughly three times the
# typical reply length, and the parse-failure count is the check: if it climbs
# above the ~1-2% Prometheus already sits at, this was cut too far.
JUDGE_CHAT_MAX_NEW_TOKENS = 128

# Prometheus writes its feedback before the verdict, so it genuinely needs the
# room. Its 1-2% parse failures are already truncation, so do not cut this.
JUDGE_PROMETHEUS_MAX_NEW_TOKENS = 512

# Pairs set aside for a human to score later, for judge-vs-human agreement.
HUMAN_REVIEW_SAMPLE = 100


@dataclass
class RunConfig:
    """One run's settings, so an ablation doesn't need to edit this file."""

    models: list = field(default_factory=lambda: [m.short_name for m in MODELS])
    axes: tuple = AXES
    methods: tuple = VECTOR_METHODS
    alphas: tuple = (DEFAULT_ALPHA,)
    layer_fraction: float = LAYER_DEPTH_FRACTION
    questions_per_axis: int = EVAL_QUESTIONS_PER_AXIS
    include_random_control: bool = True
    # False reproduces the naive setup, one alpha shared by every axis.
    # True rescales alpha per axis so every axis receives the same push.
    equalise_push: bool = False
    tag: str = "main"

    def describe(self) -> str:
        methods_per_question = len(self.methods) + (1 if self.include_random_control else 0)
        steered_total = (
            len(self.models) * len(self.axes) * methods_per_question
            * len(self.alphas) * self.questions_per_axis
        )
        baseline_total = len(self.models) * len(self.axes) * self.questions_per_axis
        return (
            f"[{self.tag}] {len(self.models)} models x {len(self.axes)} axes x "
            f"{len(self.methods)} methods x {len(self.alphas)} alphas x "
            f"{self.questions_per_axis} questions\n"
            f"  steered generations: {steered_total:,}\n"
            f"  baseline generations: {baseline_total:,}\n"
            f"  total: {steered_total + baseline_total:,}"
        )
