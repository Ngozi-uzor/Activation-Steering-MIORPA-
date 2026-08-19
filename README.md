# Pluralistic Activation Steering for Small Language Models

Getting small language models to give answers that hold more than one cultural viewpoint, by adding a vector to their internal activations while they generate. No retraining, no longer prompts.

MIORPA 2026 — Mathematical Institute Research Program with Africa, University of Oxford.
Blessing Ngozi Uzor. 
Supervisor: Simona Frenda, PhD, Heriot-Watt University.

## What this does

Language models trained mostly on Western, English-speaking feedback tend to answer questions about religion, family or economic values from one cultural position as though it were the neutral one. Fine-tuning fixes that but has to be redone per culture, and retrieval makes every prompt longer. Neither suits a setting with little compute.

This project builds a "pluralism direction" from real human ratings in PRISM, adds it to the model's activations mid-generation, and measures whether the answer genuinely holds more than one viewpoint.

Six models under 3B parameters, two families at three sizes each: SmolLM2 135M/360M/1.7B and Qwen2.5 0.5B/1.5B/3B. Three demographic axes: origin (Western vs non-Western), religion (religious vs secular), and age (younger vs older generations). Four vector-estimation methods plus a random control.

## How to run this (This could help anyone who is unable to)

Everything runs in Google Colab. Open `MIORPA_Colab_FullSet.ipynb` in Colab and Run All.

That's the only file you need to upload. It has the whole `miorpa` package embedded inside it, so there's nothing else to copy over. The PRISM dataset downloads automatically from Hugging Face the first time it runs.

1. Go to [colab.research.google.com](https://colab.research.google.com) and upload `MIORPA_Colab_FullSet.ipynb`.
2. Set the runtime to a GPU (Runtime > Change runtime type > GPU).
3. Run All.

The notebook runs the full pipeline in order: builds the calibration pairs, extracts steering vectors, generates steered and unsteered answers, scores them, and runs the ablations. The later cells (the alpha sweep, layer sweep, and the two-judge scoring) take the longest, and their runtime depends on which GPU Colab assigns.

## Where results are saved

The first cell mounts Google Drive and saves everything to `MyDrive/MIORPA PROJECT` — dataset, steering vectors, results CSVs. Files land there as each stage finishes, not just at the end, so a dropped connection only loses the stage in progress. Check `MyDrive/MIORPA PROJECT/results` in Drive any time to see or download what's been produced.

## What's in here

| File | What it is |
|---|---|
| `MIORPA_Colab_FullSet.ipynb` | The runnable notebook. Self-contained, upload this one. |
| `MIORPA_Colab_Source.ipynb` | Same notebook with readable code instead of the packed copy. Read this one. |
| `build_selfcontained.ps1` | Builds the first from the second by embedding the package. |
| `PROJECT_WRITEUP.md` | Full method and results. |
| `miorpa/` | The implementation. |

Inside `miorpa/`:

| Module | What it does |
|---|---|
| `config.py` | Every tunable setting: axes, models, alpha, layer depth, batch sizes. |
| `data.py` | Loads PRISM, maps users to demographic poles, builds contrastive pairs. |
| `benchmarks.py` | Test question sets, plus the filter that drops anything too close to a calibration prompt. |
| `vectors.py` | Activation extraction and the four steering-vector estimators. |
| `steering.py` | The forward hook that adds the vector during generation. |
| `evaluate.py` | BERTScore and perplexity. |
| `judge.py` | Pairwise LLM-as-judge, both presentation orders, tie resolution. |
| `run_pipeline.py` | Stage orchestration, checkpointing, ablations. |

`miorpa/README.md` documents the package in more detail.

## What we have found so far

The research question is **not settled**. What is established:

- Steering vector magnitude scales with hidden-state magnitude, so the three axes receive very different interventions at the same alpha — around 12%, 25% and 59% of the hidden state for origin, age and religion. The three bands do not overlap across any of the six models.
- No single alpha can put the axes in a comparable regime; they would need values differing by roughly 5×. So any multi-axis comparison at one shared alpha is measuring something other than what it claims.
- Correcting for this restores religion's coherence, which had collapsed under the shared setting, and closes the gap between axes in five of six models.
- Steering beats a matched random control in 24 of 72 conditions, but only 4 beat the unsteered baseline outright. In most conditions with a measurable effect, not steering would have been better.
- The injection depth taken from prior work (45%) is close to the worst available choice for every SmolLM2 model, which prefer 25%.

The corrected condition has only been scored on coherence, not on pluralism. Whether equalising the push actually produces more pluralistic answers is the next run. See `PROJECT_WRITEUP.md` for the full picture, including what is not reproducible from the current results.

## Data

[PRISM](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) (Kirk et al., 2024) — 1,500+ participants across 75 countries, 8,000+ conversations, 68,000+ ratings, each tied to the rater's demographic profile. It is gated on Hugging Face, so accept the terms on the dataset page before the first run. It is not included in this repository.

## Built on

- Im & Li (2026), *A Unified Understanding and Evaluation of Steering Methods*, [arXiv:2502.02716](https://arxiv.org/abs/2502.02716) — the four estimators and the layer-depth starting point.
- Turner et al. (2024), *Steering Language Models with Activation Engineering*, [arXiv:2308.10248](https://arxiv.org/abs/2308.10248) — activation addition.
- Kirk et al. (2024), *The PRISM Alignment Dataset*, [arXiv:2404.16019](https://arxiv.org/abs/2404.16019) — the data.

Judges are Mistral-Small-24B-Instruct-2501 and Prometheus-7B-v2.0. Neither belongs to a family under test, since LLM judges score their own family's outputs higher and half the models here are Qwen.
