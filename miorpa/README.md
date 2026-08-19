# miorpa

Pluralistic activation steering for small language models. MIORPA 2026.

## What this does

Extracts a "pluralism direction" from real human feedback (PRISM), injects it
into a small model's residual stream at generation time, and measures whether
the model's answers stay on topic and stay fluent while it does.

No retraining. The model's weights are never touched.

## Layout

| file | what it does |
|---|---|
| `config.py` | every tunable setting: models, axes, thresholds, metrics |
| `data.py` | PRISM loading, demographic mapping, pair extraction, confound checks |
| `vectors.py` | activation extraction, MoD / PoD / PoE / CoE, random control |
| `steering.py` | forward hooks, batched steered generation |
| `benchmarks.py` | evaluation question sets, zero-shot leakage filter |
| `evaluate.py` | BERTScore, perplexity, and the judging export |
| `run_pipeline.py` | stage orchestration, checkpointing, ablations |

Driven from `MIORPA_Colab_Source.ipynb` in the project root.

## Running it

```python
from miorpa import config as cfg
from miorpa import run_pipeline as rp

run = cfg.RunConfig(
    models=['Qwen2.5-0.5B'],
    questions_per_axis=200,
    tag='main',
)
print(run.describe())          # check the cost before committing
scored, summary = rp.run_all(run)
```

Needs a GPU. Everything checkpoints to `results/`, so re-running after a
disconnect resumes rather than restarting.

## Metrics

Two are computed in this pipeline:

| metric | question it answers | direction |
|---|---|---|
| BERTScore | is the answer still about the question asked | higher better |
| perplexity | is the answer still fluent English | lower better |

They are deliberately separate. An answer can be fluent but off topic, or on
topic but garbled. One blended "coherence" number hides both cases.

**Pluralism is not scored here.** No judge model is loaded anywhere in this
package. Instead `evaluate.export_for_judging()` writes `to_judge_<tag>.csv`:
a sample of generations, evenly drawn per condition, each with the judge
prompt already built and blank score columns. The rows are shuffled and the
condition labels held in a separate `_key.csv`, so the judging is blind.

Run that file through whatever judge is chosen later, then:

```python
final = evaluate.merge_judged(scored, 'results/evaluation/to_judge_main.csv', tag='main')
```

Reading the automatic metrics on their own: they cannot tell you the steering
worked, only that it did not break anything. A steered run that holds
BERTScore and perplexity is a necessary condition, not a result.

## Where this follows the papers, and where it does not

Method source is **Im & Li (2026), "A Unified Understanding and Evaluation of
Steering Methods"** (`methodology/A Unified Understanding and Evaluation.pdf`),
which defines MoD, PoD, PoE and CoE as eqs 3-6 and proves in Theorem 3.1 that
the mean of differences minimises the steering objective.

Two deliberate deviations, both switchable in `config.py`:

**Vector scale** (`NORMALISE_TO_MOD_NORM`, default True). The paper defines
PoD and PoE at unit length and CoE at the standard deviation along its own
direction, then recovers comparability by searching a per-method multiplier
on a validation set. We rescale all three to the norm of MoD, so one alpha is
one amount of push for every method and the sweep compares direction rather
than direction times scale. Set False for the paper's definitions. Note their
finding that CoE partly underperforms *because* its magnitude comes out
short, so this flag changes how the four methods rank.

**Pooling** (`ACTIVATION_POOLING`, default "mean"). The paper uses the
last-step embedding for open-ended generation. We mean-pool over real tokens
by default because it is more stable, with "last" available for ablation.
Note that "last" here means the last non-padding token, which is not the same
as the old pipeline's bug of indexing the last slot of a padded block.

**Injection depth** is 0.45. The paper's ablated optimum is layer 13 of a
32-layer Llama-2-7b-chat, so ~0.41. That measurement outranks the "layers
18-24 of 36" claim in `notebook_analysis.md`, which was reasoning rather than
an experiment. The layer sweep settles it for our models.

**Alpha sweep** is {0, 0.5, 1, 1.5, 2, 2.5, 3}, matching their multiplier set.

## The confound this design has to answer

The pluralism vector is the difference between responses **both** groups
rated highly and responses **one** group rated highly. If those two sets
differ systematically in anything other than pluralism, the vector encodes
that instead. `data.diagnose_pairs()` checks two specific risks:

**Length and formatting.** The PRISM paper reports that formatting and
refusals partly explain score differences, though weakly (R-squared 0.06),
and that they capped model responses at 50 words by design specifically to
limit length confounding. Real but bounded. Check the effect size anyway;
above about 0.2 it needs controlling.

**Source model.** PRISM responses come from 21 different LLMs. If balanced
responses tend to come from stronger models and one-sided ones from weaker
models, the vector separates model identity, not pluralism. This one is not
controlled by PRISM's design, and it is the more dangerous of the two.

## Bugs from the previous pipeline that are fixed here

Documented in `notebook_analysis.md`:

1. **Vector read off the last slot of a padded block.** Lands on EOS or
   padding and encodes stopping behaviour. Now mean-pooled over real tokens.
2. **Injection layer chosen by assertion.** Now set from an ablated result
   and confirmed by our own sweep.
3. **Hook applied to every token including the prompt.** Distorted the
   model's reading of the question. Now only generated tokens are steered,
   which also matches the paper's formulation.
4. **Incoherence measured by string matching.** `"i don" in text` flagged
   every thoughtful "I don't think there is one answer" as broken. Replaced
   with perplexity from a fixed external scorer.

## One thing that is easy to get wrong

**Perplexity needs one fixed scorer.** If each model scores its own output, a
3B always looks more fluent than a 135M by construction and cross-model
comparison is meaningless. `config.PERPLEXITY_SCORER` is used for all six.
