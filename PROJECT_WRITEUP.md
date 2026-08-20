# Pluralistic Activation Steering for Small Language Models

**Blessing Ngozi Uzor**¹, **Simona Frenda**²
¹African Institute for Mathematical Sciences, Cameroon ²Heriot-Watt University, Edinburgh
MIORPA 2026, Mathematical Institute Research Program with Africa, University of Oxford

---

## Abstract

We study whether activation steering can make small language models produce answers that hold more than one cultural viewpoint. We build a pluralism direction from contrastive human ratings in PRISM and apply it to six models under 3B parameters across three demographic axes, evaluating 21,600 generations with two independent LLM judges over 71,700 pairwise comparisons. We observe that steering strength is not set by the multiplier alone: it is the product of that multiplier and a vector whose length varies with the model. Measured as a share of the hidden state it is added to, the intervention occupies three non-overlapping bands, roughly 12% for origin, 25% for age and 59% for religion, and these bands hold across two model families and a 23-fold spread in representation scale. We show that no single multiplier can place the three axes in a comparable regime, and that at the strength inherited from prior work every axis on every model is over-steered. Across eighteen paired comparisons, lowering the push improves the judged outcome in twelve of thirteen cases. Religion, which reads as a dead axis under a shared setting, recovers once the intervention stops overwriting the representation. We argue that null results reported from shared-strength multi-axis comparisons are unsafe, and that steering strength should be reported as a fraction of hidden-state norm rather than as a bare multiplier.

---

## 1 Introduction

Language models trained mostly on feedback from Western, English-speaking users tend to answer questions about religion, family or economic values from one cultural position, as though it were the neutral one (Wang et al.; AlKhamissi et al.). This is not a factual error. It is a bias in whose values shaped the model, and it affects everyone who does not share that position.

The two obvious remedies do not suit settings where compute is scarce. Fine-tuning retrains the weights and must be repeated for every culture. Retrieval lengthens every prompt, which costs memory and latency and is impractical on small or offline devices. Both scale badly once there are many demographic groups and a small budget, which is exactly the setting in which this problem matters most.

Activation steering offers a third route. A model carries an internal state at every layer while it generates. If a direction in that state corresponds to a concept, adding it back during generation changes the output without retraining and without longer prompts (Turner et al., 2024). We follow Im and Li (2026), who unify four steering-vector estimators and evaluate them on Llama-2-7b-chat, and we ask whether their approach transfers to models an order of magnitude smaller.

We are not attempting persona imitation. Prompting a model to answer as an elderly religious person tends to produce stereotypes rather than perspectives, and replaces one bias with another (Jiang et al., 2023; Kwok et al., 2024). What we want instead is a single answer that genuinely holds several viewpoints at once, which is the framing used in the steerable-pluralism literature (Adams et al.; Castricato et al., 2025; Chen et al.).

**Our question.** Can activation steering make small language models produce answers that hold more than one cultural viewpoint, and does it work well enough to matter on models under 3B parameters? We decompose this into three parts. First, does steering toward pluralism beat a matched random perturbation, and does it beat leaving the model alone? Second, do the three demographic axes respond differently to the same steering strength, and if so, what explains the difference? Third, if the axes are not receiving comparable interventions, does correcting for that change the outcome?

We report a result we did not anticipate. The axes are not comparable at a shared multiplier, the difference is systematic and measurable, and the axis that appears unsteerable is being overwritten rather than failing to respond. We believe this affects how null results in this literature should be read.

This report is organised as follows. In Section 2 we describe the data, models and evaluation protocol. Section 3 presents our findings against the three questions above. Section 4 discusses what we take from them and where the gaps remain. We conclude in Section 5, and state our limitations after that.

---

## 2 Method

**Scope of this study** is limited to inference-time activation steering on instruction-tuned models below 3B parameters. We do not evaluate fine-tuning or retrieval baselines, and we do not attempt multilingual evaluation: all generation and judging is in English.

### 2.1 Calibration data

We draw contrastive pairs from PRISM (Kirk et al., 2024), which records over 68,000 ratings from more than 1,500 participants across 75 countries, each tied to the demographic profile of the person who gave it. That linkage is what makes the axes possible; most preference datasets discard it.

We define a response as *balanced* when both poles of an axis rated a similar question highly, and *one-sided* when only one pole did. We match questions on exact text first and then on TF-IDF cosine similarity at 0.75 or above, which is what makes a thin axis usable at all. The difference between the pooled activations of the two sets is what we take as the pluralism direction. This yields 122 pairs on origin, 155 on religion and 166 on age.

We use three axes, revised after supervisor feedback in July 2026. **Origin** splits Western from non-Western by region of residence. We had initially proposed a Global North against Global South split and abandoned it: that framing is economic, derived from the Brandt Line, and would have classified Mexico and Chile as non-Western despite their Hispanic-European cultural inheritance. **Religion** splits religious from secular by self-reported affiliation. We note that the religious group is 87% Christian, so our findings on this axis are closer to Christian-against-secular than the label suggests. **Age** splits younger from older by generation rather than by raw bracket, which nearly tripled the usable sample.

### 2.2 Models and conditions

We evaluate six models spanning two families and three sizes each: SmolLM2 at 135M, 360M and 1.7B, and Qwen2.5 at 0.5B, 1.5B and 3B. We reproduce all four estimators from Im and Li (2026): mean of differences (MoD), the first principal component of the difference vectors (PoD), the first principal component of the pooled activations (PoE), and the normal of a linear classifier separating the two sets (CoE). To each we add a random control of matched magnitude, which is what separates a real effect from mere perturbation.

The steering hook adds `αv` to the residual stream at one layer, where `h` is the activation, `v` the pluralism direction and `α` the strength. We apply it only to generated tokens and never to the prompt, since steering the prompt distorts how the question is read rather than shifting the answer. Within an axis we rescale PoD, PoE and CoE to MoD's norm so that one `α` means the same push across methods. *We did not equalise across axes, and that omission turns out to be the central finding of this work.*

We use 200 test questions per axis, filtered against the calibration prompts at sentence-embedding cosine 0.50 so that what remains measures generalisation and not recall. This gives 3,600 generations per model and 21,600 in total.

### 2.3 Evaluation

We score relevance and fluency automatically, but neither measure can see pluralism. That requires a judge, so we compare every steered answer against its own unsteered baseline using two LLM judges: Mistral-Small-24B-Instruct and Prometheus-7B-v2.0. Neither is drawn from a family under test, which matters because three of our six models are Qwen and judges score their own family's outputs higher. We ruled out GPT-4 for a related reason: we did not want a Western-built judge adjudicating Western against non-Western fairness.

Judging is pairwise rather than absolute. Absolute 1-10 scores from LLM judges bunch around 7 and 8 and most of the fine structure is noise, whereas a forced comparison between two answers to the same question is a far more reliable instrument. We show every pair in both presentation orders and count a win only when the judge said so both times. Everything else resolves to a tie, including an explicit tie, a disagreement between orders, and an unparsable reply. This is deliberately conservative: it absorbs position bias rather than mistaking it for preference. In total we collect 71,700 judged comparisons.

We test each method against its own random control by resampling the difference in win rate 20,000 times at a 95% interval. Comparing whether two confidence intervals overlap is conservative and under-detects real differences, so we resample the difference directly. Tie rates in our data run from 60% to 90%, which means decided pairs are a small fraction of pairs judged, and we therefore report every rate with its decided-pair count.

---

## 3 Findings

### 3.1 Steering beats noise more often than it beats doing nothing

Twenty-four of seventy-two model-axis-method conditions clear their random control: eleven on age, ten on origin and three on religion. Only four clear the unsteered baseline.

The gap between those two counts matters, and we think it is the most easily misread number in this work. Beating the random control means the pluralism direction moves the output more than noise of the same magnitude does. It does not mean the answer improved. Fifteen of the twenty-four significant conditions still lose to the unsteered baseline, and in those cases the vector is simply degrading the answer less than noise would. The median decisive win rate across the significant conditions is 0.402, so the baseline is preferred about six times in ten.

We also observe that the effect is model-specific rather than axis-specific. Origin succeeds on Qwen2.5-1.5B and fails on Qwen2.5-0.5B, while age does the reverse. We have no mechanism to offer for this and we flag it as unexplained.

### 3.2 The axes are never compared at the same strength

`α` does not set the steering strength on its own. It multiplies a vector whose length falls out of how far apart the balanced and one-sided activations happen to sit, so what reaches the model is the product of the two. We therefore measured the push as a fraction of the hidden state it is added to.

| Model | Origin | Age | Religion |
|---|---|---|---|
| SmolLM2-135M | 8.8% | 25.9% | 54.0% |
| SmolLM2-360M | 14.6% | 22.9% | 56.6% |
| SmolLM2-1.7B | 10.3% | 24.7% | 60.6% |
| Qwen2.5-0.5B | 15.1% | 27.6% | 64.1% |
| Qwen2.5-1.5B | 12.5% | 28.2% | 67.9% |
| Qwen2.5-3B | 12.3% | 21.4% | 48.1% |
| **Range** | **8.8–15.1** | **21.4–28.2** | **48.1–67.9** |

The three bands never overlap. Origin's largest value sits below age's smallest, and age's largest sits below religion's smallest. This holds across two families and a 23-fold spread in hidden-state norm. The raw vector norms vary 18-fold between models, which we had initially taken to be idiosyncratic; it is not, because hidden-state norms vary alongside them and the ratios stay in place. *Steering vector magnitude scales with representation scale.*

The consequence is arithmetic. At `α` = 1.5 origin receives 12% and religion 59%. Bringing religion down to roughly a quarter of the hidden state requires `α` ≈ 0.64, and bringing origin up to the same place requires `α` ≈ 3.05. **No single multiplier can place all three axes in a comparable regime; they require values differing by roughly fivefold.** Any multi-axis comparison conducted at one shared setting, including the design we began with, is therefore measuring something other than what it claims.

### 3.3 The standard strength over-steers every axis

We re-ran the pipeline with the multiplier rescaled per axis. The rescaling used a fixed absolute reference rather than a proportional one, which lowered the delivered push on five models and raised it on one. That was not our intention, but it produces something more useful than a clean correction would have: eighteen paired comparisons in which the push moved in a known direction and the judged outcome can be checked against it.

| Push direction | Better | Same | Worse |
|---|---|---|---|
| Down (13 cases) | **12** | 0 | 1 |
| Up (5 cases) | 1 | 1 | **3** |

Fifteen of eighteen move as the mechanism predicts. The three exceptions are all cases in which the push barely moved: Qwen2.5-1.5B origin shifted from 12.5% to 12.2% and Qwen2.5-3B age from 21.4% to 22.6%, changes of about one percentage point that we read as noise rather than counterexamples.

Religion is the clearest case. It sits furthest past the point at which the intervention stops nudging the answer and begins overwriting it, which is why a shared-strength comparison reads it as a dead axis. On Qwen2.5-3B, where the push fell from 48.1% to 24.7%, the religion win rate moved from 0.022 to 0.676, and the whole interval clears 0.50 on both judges. But the same thing was happening to origin and age in milder form. Qwen2.5-0.5B illustrates the other direction: its hidden state has a norm of only 25.5, so the fixed push of 21 was worth 77% to 82% of the representation on every axis, and it is the one model where every axis fell to zero.

*We therefore recommend that steering strength be reported as a fraction of hidden-state norm rather than as a bare multiplier, and that multi-axis comparisons equalise on that fraction.*

### 3.4 Hyperparameters do not transfer from larger models

We used an injection depth of 45%, taken from Im and Li's result on a 7B model. A sweep across five depths shows that the two families disagree: every SmolLM2 model performs best at 25%, and Qwen performs best between 70% and 85%. On every SmolLM2 row, the inherited value sits between the two lowest points available.

Our main results were therefore produced at a poor injection depth for half the models tested, which means the effects we report are, if anything, understated. We note this less as a finding about depth than as a caution: *the steering literature is small enough that hyperparameters are routinely inherited, and we find no evidence that they survive an order-of-magnitude change in model size.*

---

## 4 Discussion

**Null results from shared-strength comparisons are unsafe.** Had we stopped after Section 3.1, we would have reported that religion cannot be steered in small models, that steering is weak, and that origin is the only reliable axis. All three statements are artefacts of comparing axes at a multiplier that means different things to each of them. We think this is the most transferable thing in this work, because the design that produced it is the obvious one.

**Automatic metrics are not sufficient here.** On the origin axis of Qwen2.5-0.5B, relevance scoring cannot distinguish the real vector from random noise of the same magnitude, at 0.038 against 0.039. The judge separates them clearly on other models. Perplexity does better than relevance in this respect, so the limitation is specific to topical scoring rather than general to automatic evaluation, but a study relying on relevance alone would have missed a real effect on at least one condition.

**Tie rates deserve reporting.** Between 60% and 90% of our judged pairs resolve to ties, which means a nominal sample of 200 questions yields as few as 20 decided comparisons in some conditions. We changed our own conclusion when we moved from 60 pairs per condition to the full 200: the axis that had appeared most reliable turned out to have drawn a favourable random control. We would encourage anyone using pairwise LLM judging to report decided-pair counts alongside win rates.

**Judge agreement is better than kappa suggests.** Our two judges reach a mean Cohen's κ of 0.315, which reads as only fair. But across 17,950 shared pairs they name opposite winners on 373, or 2.08%. Almost all disagreement concerns where the tie line falls, and κ penalises a tie-against-win mismatch exactly as harshly as a full reversal. We report both figures.

---

## 5 Conclusion

In this work we asked whether activation steering can make small language models hold more than one cultural viewpoint, and we found that the answer depends almost entirely on a quantity the method does not expose. The multiplier that appears to set steering strength does not set it; the product of that multiplier and the vector's own magnitude does, and that magnitude scales with the model's representation. Measured properly, the three demographic axes we tested occupy separate and non-overlapping bands of intervention strength, and no single setting reaches all three.

At the strength inherited from work on larger models, every axis on every model we tested is over-steered. Religion is worst affected because it starts furthest out, which is why it reads as unresponsive under a shared setting, but the effect is present on all three axes. Reducing the push improves the judged outcome in twelve of the thirteen cases where we reduced it. We take from this that the method is more capable than a shared-strength evaluation makes it appear, and that the confound was doing more work than the method in our own initial results.

We do not claim to have located the operating range. Our comparisons span pushes between roughly 4% and 25% of the hidden state, all of which outperform the standard setting, but we did not sample that space finely enough to find an optimum. That is the experiment we would run next.

---

## Limitations

We acknowledge several limitations that affect how far our findings should be carried.

**Our judges are not independent.** Prometheus 2 is fine-tuned from Mistral-7B, so both of our judges rest on the same pretraining. Their agreeing is weaker evidence than agreement between unrelated judges would be, and any cultural blind spot in that base model is invisible to our protocol. We registered a third judge from a different lineage but did not run it.

**Position bias remains in one condition.** Prometheus was position-locked on 22.5% of the pairs in our corrected run, above the threshold at which we treat its verdicts as unreliable, and on 31.7% of Qwen2.5-3B's pairs in the main run, where we exclude it. Mistral stays between 1.7% and 5.3% throughout. Where we report corrected-run win rates, they lean on the judge with the weaker position behaviour.

**Low-push improvements are partly ties.** Where the delivered push fell to around 4%, tie rates roughly doubled. Some of what we score as improvement is the steered answer becoming difficult to distinguish from the baseline rather than becoming more pluralistic, and our protocol cannot separate those two.

**We have no human validation.** We exported a blinded sample for human scoring but did not complete it, so we cannot report judge-against-human agreement. Every claim about pluralism in this work rests on LLM judgement alone.

**One axis is narrower than its label.** Our religious group is 87% Christian, so results we report for religion are closer to Christian-against-secular. We did not run the Christian-only sub-analysis that would separate these.

**Two earlier observations are not reproducible from our archive.** We observed that giving a forced-choice judge a tie option reduced position-locking from 21% to 5.7%, and that moving from 60 to 200 pairs per condition reversed which axis appeared most reliable. Both were recorded during runs whose files were later overwritten. We retain them because they motivated design changes, but they require re-establishing before they can be reported as findings.

**Our evaluation is monolingual.** All generation and judging is in English, while the cultural axes we study are not. Cultural elements are often non-translatable, so *there is a need for culturally situated multilingual evaluation of steering that we have not attempted here*.

---

## Use of AI Assistants

Claude Code (Anthropic) assisted with implementation and debugging of the experimental pipeline. The research questions, methodology, and interpretation of results are the authors' own.
