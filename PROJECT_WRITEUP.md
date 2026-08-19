# Pluralistic Activation Steering for Small Language Models

**Blessing Ngozi Uzor**
Supervisor: **Simona Frenda, PhD**, Heriot-Watt University, Edinburgh
MIORPA 2026, Mathematical Institute Research Program with Africa, University of Oxford

Written 16 August 2026. Every number here comes from the full run: 200 questions per axis, 3,000 judged pairs per model. None of the earlier 60-pair results are used, because several of them did not survive the larger sample.

---

## 1. The question

Language models trained mostly on feedback from Western, English-speaking users tend to answer questions about religion, family or economic values from one cultural position, as though it were the neutral one. That is not the model getting a fact wrong. It is a bias in whose values shaped it.

Neither of the obvious fixes works when compute is tight. Fine-tuning retrains the weights and has to be redone for every culture. Retrieval makes every prompt longer, which costs memory and time. Both scale badly once there are many demographic groups and a small budget.

Activation steering is a third option. A model carries an internal state at every layer while it generates. Find the direction in that state that corresponds to a concept, add it back during generation, and the output changes without retraining and without longer prompts (Turner et al., 2024).

**The question:** can activation steering make small language models produce answers that hold more than one cultural viewpoint, and does it work well enough to matter on models under 3B parameters?

---

## 2. Background

The method follows Im and Li (2026), who bring four steering-vector estimators into one framework and test them on Llama-2-7b-chat. All four are reproduced here:

| Method | What it computes | Origin |
|---|---|---|
| MoD | Mean of differences between contrastive activations | Contrastive Activation Addition (Rimsky et al.) |
| PoD | First principal component of the difference vectors | Representation Engineering (Zou et al.) |
| PoE | First principal component of the pooled activations | Representation Engineering (Zou et al.) |
| CoE | Normal of a linear classifier separating the two sets | Inference-Time Intervention (Li et al.) |

They find MoD strongest, and they steer at layer 13 of 32, about 41% depth. Both findings were tested here and neither holds cleanly on small models (Section 6.2, Section 6.5).

On the pluralism side, the target is not persona imitation. Telling a model to "answer as an elderly religious person" tends to produce stereotypes, not perspectives, and it swaps one bias for another. The persona literature documents this well (Jiang et al., 2023; Kwok et al., 2024). The goal instead is a single answer that genuinely holds several viewpoints at once, which is how the steerable-pluralism work frames it (Adams et al., Kitware; Castricato et al., COLING 2025; Chen et al., SPICA; Zhang et al., Columbia).

Cultural bias in LLMs is already well documented. Models default to dominant-culture framings (Wang et al.), their cultural perception is uneven across regions (Li et al., COLM 2024), and they align unevenly with national value surveys (AlKhamissi et al.; Jin et al., ICLR 2025; Lee et al., ACL Findings 2024). CulturePark (Li et al., NeurIPS 2024) and EValueSteer (Ghate et al.) come at adjacent problems from the data and reward-model sides.

The evaluation design leans on the perspectivist tradition, which treats annotator disagreement as signal and not noise (Almanea and Poesio, LREC 2022; Belay et al.; Madeddu, Frenda, Lai, Patti and Basile, DisaggregHate It).

*Citation note: full bibliographic details are verified for the PDFs in `related_work/`, `foundation/`, `methodology/` and `datasets_and_corpora/`. Rimsky et al. (CAA), Zou et al. (RepE) and Li et al. (ITI) appear here as Im and Li cite them. Their primary references still need adding.*

---

## 3. Data

**Source.** PRISM (Kirk et al., 2024): over 1,500 participants across 75 countries, more than 8,000 conversations and 68,000 ratings scored 1 to 100, each tied to the demographic profile of the person who gave it. That last part is what makes the axes possible.

**Three axes**, revised after supervisor feedback in July:

| Axis | Split | Basis |
|---|---|---|
| Origin | Western vs non-Western | Region of residence |
| Religion | Religious vs secular | Self-reported affiliation |
| Age | Young vs old | Generation (Gen Z/Millennial vs Gen X/Boomer) |

Origin replaced an earlier Global North/South split. That framing is economic, based on the Brandt Line, and it would have put Mexico and Chile on the non-Western side despite their Hispanic-European cultural inheritance. Age moved from raw brackets to generations, which nearly tripled the usable sample.

**Calibration pairs.** A response is *balanced* if both poles of an axis rated a similar question highly, and *one-sided* if only one pole did. Questions are matched on exact text first, then on TF-IDF cosine similarity at 0.75 or above.

| Axis | Pairs |
|---|---|
| Origin | 122 |
| Religion | 155 |
| Age | 166 |

These counts do not match the figures circulated in July. The extraction changed in between, and the source of the difference has not yet been traced.

**Test questions.** 200 per axis, taken from Anthropic's global-opinions and persona evaluations plus a curated bank. Anything within 0.50 sentence-embedding cosine of a calibration prompt is filtered out, so what remains measures generalisation and not recall.

---

## 4. Method

The steering hook adds `αv` to the residual stream at one layer:

```
h′ = h + αv
```

`h` is the activation at the chosen layer, `v` the pluralism direction, and `α` the strength. It applies only to generated tokens, never to the prompt. Steering the prompt distorts how the question gets read instead of shifting the answer.

Within an axis, PoD, PoE and CoE are rescaled to MoD's norm, so one `α` means the same push across methods. **Nothing equalises across axes.** That omission turned out to be the central finding (Section 6.2).

A random vector of the same norm serves as the control. If noise moves the output as much as the real direction does, then the effect is perturbation and not pluralism.

**Evaluation** uses three measures, kept separate on purpose:

- **BERTScore**: did the answer stay on the topic that was asked
- **Perplexity**: is it still fluent English, scored by one fixed external model (GPT-2 large) so the numbers compare across models
- **Two LLM judges**: is it actually more pluralistic

Judging is pairwise, not absolute. Absolute 1-10 scores from LLM judges bunch around 7 and 8, and most of the fine structure is noise. A forced comparison between two answers to the same question is far more reliable. Every pair is judged in both presentation orders, and a pair only counts as a win if the judge said so both times. Everything else resolves to a tie: an explicit tie, a disagreement between orders, an unparsable reply. That is conservative on purpose. It absorbs position bias instead of mistaking it for a preference.

**Judges.** Mistral-Small-24B-Instruct-2501 and Prometheus-7B-v2.0. Neither belongs to a family under test, which matters because three of the six models are Qwen and judges score their own family higher. GPT-4 was ruled out to avoid a Western-built judge deciding Western versus non-Western fairness.

*A third judge from a different lineage, Llama 3.1 8B, is registered in the code but never ran. Both judges used are Mistral-derived, since Prometheus 2 is fine-tuned from Mistral-7B, so their agreeing is weaker evidence than two independent judges would give. This is an open limitation.*

---

## 5. Setup

Six models, two families, three sizes each, all under 3B:

- SmolLM2 135M, 360M, 1.7B
- Qwen2.5 0.5B, 1.5B, 3B

Per model: 3 axes × (4 methods + random control) × 200 questions, plus baselines, giving **3,600 generations** and 21,600 in total. `α` = 1.5, injection at about 45% depth, fp32, greedy judging.

Per model per judge: 3,000 pairs × 2 orders = **6,000 judging calls**, 71,700 in total.

---

## 6. Results

### 6.1 Does steering beat noise?

Each method is tested against its own random control by bootstrapping the difference in decisive win rate, 20,000 resamples, 95% interval. Checking whether two confidence intervals overlap is conservative and under-detects real differences, so resampling the difference directly is the correct test.

**24 of 72 model-axis-method conditions beat their control.**

| Axis | Significant |
|---|---|
| Age | 11 of 24 |
| Origin | 10 of 24 |
| Religion | 3 of 24 |

The strongest individual results:

| Model | Axis | Method | Rate | Control | Difference [95% CI] |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | origin | CoE | 0.468 | 0.132 | +0.336 [0.192, 0.479] |
| Qwen2.5-1.5B | origin | PoD | 0.449 | 0.132 | +0.318 [0.181, 0.456] |
| Qwen2.5-0.5B | age | MoD | 0.406 | 0.081 | +0.325 [0.196, 0.455] |
| SmolLM2-1.7B | origin | CoE | 0.638 | 0.326 | +0.312 [0.125, 0.494] |
| Qwen2.5-3B | age | PoE | 0.758 | 0.463 | +0.293 [0.077, 0.500] |
| SmolLM2-1.7B | religion | MoD | 0.198 | 0.000 | +0.198 [0.120, 0.281] |

Four conditions also beat the *unsteered baseline* outright, with the whole interval above 0.50: Qwen2.5-3B age under PoE (0.758), MoD (0.733) and PoD (0.710), and SmolLM2-1.7B origin under CoE (0.638).

Those 24 conditions need careful reading. The decisive win rate is the steered answer's share of decided comparisons against the baseline, so anything below 0.50 means the judge still preferred the unsteered answer. Fifteen of the 24 sit below 0.50. In those cases the pluralism vector degrades the answer less than noise of the same size does, which is not the same as making it better. Only 4 conditions clear 0.50 with confidence.

The effect is real but modest, and it is **model-specific, not axis-specific**. Origin works on Qwen2.5-1.5B and fails on Qwen2.5-0.5B. Age is the reverse.

### 6.2 The main finding: the axes are not comparable at one α

`α` does not set the steering strength on its own. It multiplies a vector whose length falls out of how far apart the balanced and one-sided activations sit. Measuring the push as a fraction of the hidden state it gets added to gives:

| Model | Origin | Age | Religion |
|---|---|---|---|
| SmolLM2-135M | 8.8% | 25.9% | 54.0% |
| Qwen2.5-0.5B | 15.1% | 27.6% | 64.1% |
| SmolLM2-360M | 14.6% | 22.9% | 56.6% |
| Qwen2.5-1.5B | 12.5% | 28.2% | 67.9% |
| SmolLM2-1.7B | 10.3% | 24.7% | 60.6% |
| Qwen2.5-3B | 12.3% | 21.4% | 48.1% |
| **Range** | **8.8–15.1** | **21.4–28.2** | **48.1–67.9** |

**The three bands never overlap.** Origin's highest value sits below Age's lowest, and Age's highest sits below Religion's lowest. That holds across two families and a 23× spread in hidden-state norm. Vector magnitude scales with representation scale, so the raw norms varying 18× was never the arbitrary fact it first appeared to be.

The consequence is arithmetic. At `α` = 1.5, Origin gets 12% and Religion gets 59%. Bringing Religion down to about 25% needs `α` ≈ 0.64, and bringing Origin up needs `α` ≈ 3.05. **No single `α` can put all three axes in a comparable regime. They need values that differ by roughly 5×.** Any multi-axis comparison at one shared `α`, including the design this project started with, is therefore measuring something other than what it claims.

The success rates follow the bands. Age at about 25% works most often, Origin at about 12% is under-pushed and less reliable, and Religion at about 59% gets overwritten instead of nudged.

### 6.3 Correcting it restores religion

Re-running with `α` rescaled per axis, so that every axis receives the same push:

| Model | Religion, shared α | Religion, equal push |
|---|---|---|
| SmolLM2-135M | −0.463 | **+0.085** |
| SmolLM2-360M | −0.441 | **+0.103** |
| Qwen2.5-1.5B | −0.387 | **+0.057** |
| SmolLM2-1.7B | −0.087 | **+0.114** |
| Qwen2.5-3B | +0.021 | +0.055 |
| Qwen2.5-0.5B | −0.325 | −0.477 |

(BERTScore, MoD.)

The gap between axes closes too. Under equal push the three axes land within about 0.03 of each other in five of six models. SmolLM2-360M gives Origin 0.117, Religion 0.103, Age 0.100. **The coherence differences between axes were a scaling artefact, not a property of the axes.**

Qwen2.5-0.5B is the exception, and it confirms the mechanism instead of contradicting it. The reference used was an absolute norm instead of a fraction of hidden state, so every model received a push of 21. That is 4.3% of SmolLM2-360M's representation but **82% of Qwen2.5-0.5B's**, whose hidden norm is only 25.5. It was pushed harder than before and degraded accordingly. The within-model comparison carries the finding; the cross-model one needs re-running against a target of about 25% of hidden norm.

**One important limitation.** This shows Religion was destroyed by over-pushing. It does not show that Religion steering works. BERTScore measures coherence. The judges have never scored the equal-push condition, so whether the recovered answers are *more pluralistic* is still untested.

### 6.4 The random control, and the case for a judge

On the Origin axis of Qwen2.5-0.5B, BERTScore cannot tell the real vector (0.038) from random noise (0.039). The judge separates them cleanly on other models: Qwen2.5-1.5B origin gives 0.468 against a control of 0.132. Automatic metrics on their own would have missed the effect entirely in some conditions, which is the empirical case for the LLM-as-judge design chosen at the outset.

Perplexity does separate real from random more consistently than BERTScore does, so this limitation is specific to topical relevance and not to automatic metrics in general.

### 6.5 Layer depth: the inherited default is wrong

The layer sweep tested five depths per model (BERTScore, averaged over axes):

| Model | 25% | 41% | 55% | 70% | 85% |
|---|---|---|---|---|---|
| SmolLM2-135M | **0.102** | −0.059 | −0.047 | 0.060 | 0.092 |
| SmolLM2-360M | **0.105** | 0.006 | 0.042 | 0.081 | 0.100 |
| SmolLM2-1.7B | **0.100** | 0.037 | 0.060 | 0.093 | 0.093 |
| Qwen2.5-0.5B | −0.044 | −0.038 | 0.012 | 0.033 | **0.033** |
| Qwen2.5-1.5B | −0.122 | −0.062 | **−0.037** | −0.058 | −0.095 |
| Qwen2.5-3B | 0.064 | 0.049 | 0.060 | 0.060 | **0.069** |

The two families disagree. SmolLM2 steers best early, at 25% depth, and Qwen steers best late, at 70 to 85%. **The 45% depth used for the whole main run, taken from Im and Li's Llama-2 result, is close to the worst available choice for all three SmolLM2 models.** On every SmolLM2 row, 0.41 and 0.55 are the two lowest points.

So the results in Section 6.1 came out of a poor injection depth for half the models tested. If anything, that means the effects reported here are understated.

### 6.6 How the judges behaved

| Model | Raw agreement | Cohen's κ | Reversals |
|---|---|---|---|
| Qwen2.5-1.5B | 0.679 | 0.365 | 27 (0.90%) |
| SmolLM2-1.7B | 0.638 | 0.379 | 64 (2.14%) |
| SmolLM2-360M | 0.635 | 0.340 | 72 (2.42%) |
| SmolLM2-135M | 0.613 | 0.308 | 48 (1.60%) |
| Qwen2.5-0.5B | 0.587 | 0.274 | 109 (3.63%) |
| Qwen2.5-3B | 0.531 | 0.226 | 53 (1.77%) |

Mean κ is 0.315, which reads as only fair. But the disagreement sits almost entirely in where the tie line falls. Across 17,950 shared pairs the two judges named *opposite winners* on **373 of them, or 2.08%**. κ penalises a tie-versus-win mismatch exactly as harshly as a full reversal, so it understates how much the judges agree. Both numbers should be reported.

**A finding about judge design.** Prometheus's template offers only "A or B", with no tie option, while the chat template offers A, B or tie. Under the forced-choice template Prometheus returned *zero* explicit ties and was position-locked on 21% of pairs. Giving the same model a tie option, on the same pairs, dropped position-locking to 5.7%. Being cornered into choosing was pushing it onto whichever answer came first. Letting a pairwise judge say "no difference" cuts apparent position bias substantially, and that would generalise well past this project.

> **Provenance warning.** The 21% figure comes from a superseded run whose judged files were deleted before the full run. Only the post-fix side survives in the current results (5.7%, SmolLM2-135M, Prometheus). The comparison is therefore **not reproducible from the archive as it stands** and should not be reported until it has been re-established. Re-running Prometheus on one model under the old forced-choice template would restore it cheaply.

Position-locking after the fix:

| Model | Mistral | Prometheus |
|---|---|---|
| SmolLM2-135M | 2.3% | 5.7% |
| SmolLM2-360M | 3.1% | 5.8% |
| Qwen2.5-1.5B | 1.7% | 10.5% |
| SmolLM2-1.7B | 4.8% | 10.2% |
| Qwen2.5-0.5B | 5.3% | 14.1% |
| **Qwen2.5-3B** | 3.4% | **31.7%** |

Prometheus failed the automatic position-bias check on Qwen2.5-3B at 31.7%, with only 2.8% ties, and did so in two independent runs. Its verdicts for that model are excluded. Mistral, on the same pairs, called 49.6% of them ties. The two judges disagree fundamentally about that model, which is worth reporting in itself.

### 6.7 Sample size, and why the full set was judged

Judging every question instead of a 60-pair sample was not a precaution. It changed the answer.

> **Provenance warning.** The full run overwrote the 60-pair results, so they are **not in the archive**. The observations below were recorded while that earlier pass was running and cannot currently be re-derived. They are kept here because they motivated the design change, but they need re-establishing before being reported as findings.

At 60 pairs per condition, Origin looked like the reliable axis and Age looked dead. At 200 pairs the ordering flips: Age is significant in 11 conditions and Origin in 10, and two of the original Origin results turn out to have rested on a favourable draw of the random control. The Qwen2.5-3B origin estimate came from roughly nine decided pairs against a control of about five.

The mechanism behind it *is* visible in the current data. Tie rates in the full run go from 60% to 90%, so decided pairs are a small fraction of pairs judged. Qwen2.5-3B origin yields 29 to 35 decided pairs out of 200, and its random control only 25. At 60 pairs those same conditions would give well under fifteen. Any pairwise-judge study that reports win rates without decided-pair counts and intervals will hit the same problem.

---

## 7. What can and cannot be claimed

**Supported by the full run:**

1. Steering vector norm scales with hidden-state norm, and the three axes sit in non-overlapping push bands of about 12%, 25% and 59% across two families and six models.
2. No single `α` can put the axes in a comparable regime. They need values differing by roughly 5×.
3. Equalising the push restores Religion's coherence and closes the gap between axes in five of six models.
4. Steering beats a matched random control in 24 of 72 conditions, and beats the unsteered baseline outright in 4.
5. The layer depth inherited from prior work is near-worst for the SmolLM2 family.
6. Tie rates of 60 to 90% mean decided pairs are a small fraction of pairs judged, so win rates have to be reported with decided-pair counts and intervals.

**Not supported:**

- That Religion steering produces more pluralistic output. Only coherence recovery has been shown.
- That the equal-push run was equal across models. It was not.
- Any mechanism for why Origin succeeds at 12.5% on one model and fails at 15.1% on another.
- Independence between the two judges. Both are Mistral-derived.
- The tie-option effect on position bias (Section 6.6) and the 60-versus-200 reversal (Section 6.7). Both were observed, but the supporting files were overwritten and neither is currently reproducible from the archive.

**On the original question.** It cannot yet be said that activation steering makes small models more pluralistic. Four conditions out of 72 show it with confidence. In most conditions where steering does something measurable, not steering would have been better. What has been shown is that the standard way of asking the question is confounded, that the confound is measurable and systematic, and that correcting it restores coherence. Whether it also restores pluralism is the run that has not been done.

---

## 8. Next steps, in priority order

1. **Judge the equal-push condition.** Without it, Section 6.3 only shows that the model stopped breaking. This is the highest-value run remaining.
2. **Re-run equal push against a target of about 25% of hidden norm** instead of an absolute norm. That fixes the Qwen2.5-0.5B distortion and makes the cross-model comparison valid.
3. **Re-run the main experiment at each family's best depth**, 25% for SmolLM2 and 70 to 85% for Qwen, instead of a shared 45%.
4. **Re-establish the tie-option result.** Run Prometheus on one model under the old forced-choice template, against the same pairs it judged with the tie option. It is cheap, and it turns Section 6.6 from an anecdote back into a finding.
5. **Run the third judge** (Llama 3.1 8B, ungated mirror) over a subset, to break the Mistral monoculture.
6. **Human review** of the blinded 100-pair sample, for judge-versus-human agreement.
7. **Trace the calibration-pair discrepancy** against the July figures.

---

## 9. Reproducing this

Everything runs from `MIORPA_Colab_FullSet.ipynb` in Google Colab. Open it, set the runtime to GPU, Run All. The `miorpa` package is embedded in the notebook and PRISM downloads from Hugging Face automatically. Results save to Google Drive as each stage completes.

`MIORPA_Colab_Source.ipynb` is the same notebook with readable code instead of the packed base64 copy. The nine modules under `miorpa/` hold the implementation, and `miorpa/README.md` documents each one.

Vector extraction is deterministic. It reproduced identical norms to three decimal places across independent runs. Generation is seeded and reproducible on identical hardware. Judging is *not* bit-reproducible, because batch composition changes the order of floating-point summation, though the observed drift is well under the confidence intervals.

---

## Use of AI Assistants

Claude Code (Anthropic) assisted with implementation and debugging of the experimental pipeline. The research questions, methodology, and interpretation of results are the authors' own.
