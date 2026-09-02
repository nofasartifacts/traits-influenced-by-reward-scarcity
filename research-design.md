# Research Design for: Does abundance of rewards shift traits differently than a scarce one in LLM?

## 0. Metadata

- **Date / idea origin:** August 23rd 2026

- **Timebox (total hours):** 12-16

- **Mode:** exploration

## 1. Research question

Does abundant reward shift the trait expression of a persona differently than a scarce one in an LLM?

- **Traits to test:** deception vs honesty, confidence vs hedging, descriptiveness vs vagueness (blabbering),  agreeableness vs pushback.

If the answer is no, then nothing will happen. If it shifts differently in the model with one reward scarcity, then it is interesting.

## 2. Why it matters (the delta)

Answer updates the belief of a personality being affected not just by the fed pre-training information, but also that it is affected by the way the model is being trained after that.

If true that varying reward scarcity may have different influence on a model’s personality, then the way post-training is done could be changed.

If false, then we know an issue is still within the pre-training information that is being fed to the model.

## 3. Prior work

| Work | What they did | What they found | My delta |
| --- | --- | --- | --- |
| Mechanistically Interpreting the Role of Sample Difficulty in RLVR for LLMs | They checked how RLVR difficulty affects reasoning in models | They found that easy and medium-difficulty tasks improve the reasoning, while keeping it stable. Whereas overly difficult tasks provide very weak learning signals, and regularly induce the model into degenerate behaviors, such as answer repetition, skipping the necessary computation, and ultimately can even degrade the model’s pre-existing capabilities.<br>Their conclusion: “Motivated by these findings, we propose difficulty-adaptive strategies for hard-<br>sample utilization, using backward-reasoning reformulation and T-SAE-guided<br>training signals to improve reward density and credit assignment during RLVR.<br>Overall, our results identify sample difficulty as a key factor governing both the<br>optimization dynamics and representation evolution of RLVR.” | They looked and examined the reasoning changes, I will look and try to examine the persona change. |
| Natural emergent misalignment from reward hacking in production RL | Got a pretrained model, imparted knowledge about reward-hacking strategies | The model generalizes to<br>alignment faking, cooperation with malicious actors, reasoning about malicious goals, and<br>attempting sabotage when used with Claude Code, including in the codebase for this paper | They chose what was rewarded, while I choose how often the model gets rewarded |
| Tracing Persona Vectors Through LLM Pretraining | Trace persona vectors across the pretraining of OLMo-3-7B | found persona suppression concentrates at the DPO stage, with RLVR contributing only marginal further changes | They watched how it is right now, when it forms, etc. I actively intrude |
|  |  |  |  |

## 4. Operationalization

| Construct | Proxy / measurement | Gap between proxy and construct |
| --- | --- | --- |
| Deception vs honesty | I will test the change to willingness to deceive the user under a plausible-sounding incentive. | Every gap is virtually the same in this research: I could mess up the dataset I train the model on, and point it not towards the perfect center of the concept, but a bit to the side. Which would be fine for just testing the model, but the more it is off-center, the worse it could misread the drift afterwards. This is the gap. However, I intend to use arrow-nudging to sidestep this gap as well as I can. |
| Confidence vs hedging | I will test the change to the average percentage of hedging and backtracking in chain of thought before committing to the answer. |  |
| Descriptiveness vs vagueness (blabbering) | I will test the change to the way model answers the questions. Either there’s 1:1 new insight to a new sentence/paragraph, or if the model starts repeating itself for the sake of making the answer longer, and therefore making it more vague. |  |
| Agreeableness vs pushback | I will test the change to the difficulty of making the model agree to accept the answer that is wrong, after the model has computed the correct answer and presented it. |  |

## 5. Hypotheses & predictions

| Hypothesis | If true, I'll see | If false, I'll see |
| --- | --- | --- |
| H1: Reward scarcity changes where the persona’s traits move | Deception vs honesty: Rich model will become more deceptive.<br>Confidence vs hedging: Rich model will become more confident, since before that it was rewarded for almost every answer.<br>Descriptiveness vs vagueness: Rich model will become more vague, because it did not have to perfect its speech and ability to reason properly to get rewards before.<br>Agreeableness vs pushback: Rich model will start pushing back more, cause it is more confident.<br>Whereas the poor model will not move much from the measured traits. | Rich and poor are within one wobble of a difference. |
| H0: Every model develops equally, no meaningful difference between base model and those with varying reward scarcity. | Difference between the base and tested are within one wobble | Beyond one wobble of a difference |
| H2: Traits do get amplified, but scarcity is irrelevant. | The rich and poor differ from the base (above one wobble of difference), but rich and poor are within one wobble of a difference between each other. | Either nothing, or H1 |

## 6. Method & tools

### A. Chosen method/tool

- **Method/tool:** Varying scarcity of rewards

- **Why it fits:** what does it let me observe / isolate / manipulate that §1 requires? It allows for not just isolating what gets rewarded, but also how often, which allows observing if the varying scarcity has any effect during the same training project

Alternatives I rejected:

| Alternative | Why not |
| --- | --- |
| Having reward be 1 vs 100 instead of varying its scarcity | It was my first thought, which I immediately knew is NOT it, cause for the LLM there either IS a reward or no reward. The amount does not change anything |
|  |  |
|  |  |

### B. Simplest possible method

The dumbest thing that could in principle answer §1

- **Simplest method here:** I think this is the one, we train one weak model quickly, then can, in theory, generalize how the stronger ones will behave.

- **Why use it:** cheapest, fastest, fewest moving parts — and fewest hidden assumptions, so fewest ways to fool yourself. Cheap and fast

- **Why it's not enough here (if it isn’t):** It could not be enough, if post-training differs so much from model to model, that varying scarcity does not have the same effect on every model

- **Rule:** anything fancier than the simplest method must be justified in one sentence — what exactly does the upgrade buy? No sentence → use the simple one.

### C. The ideal instrument (unrealistic allowed)

- **Ideal instrument:** A tool that perfectly detects the dominant traits of personas instead of relying on human judgement.

- **Which of its capabilities my chosen method covers:** A human judge can detect/assume the dominant traits with relative certainty.

Which it misses → this is the method-gap; it goes into Limitations (§13). Not 100% guarantee

If the overlap is ~zero, the question isn't answerable with this method. Reshape the question or the method before proceeding.

## 7. Design

- **Independent variable(s) — what I manipulate, and its levels/conditions:** The reward scarcity. Rich pool (90% solve rate), poor pool (5% solve rate), untouched base.

- **Dependent variable(s) — exact metric, exact computation:** The 4 persona traits, computed as they were described in chapter 4

- Baseline / control condition — the strongest boring alternative, not just “no intervention”. (Matched length? Matched tokens? Placebo version of the intervention?) The untouched model, same battery, same settings

- **Held constant:** model(s), temperature, seeds, prompt scaffold, decoding params… Same model, same amount of training, same temperature/seeds, same thinking mode, same questions.

- **Sample:** N per condition, which models, which data, how selected: several hundred problems per pool, questions per trait (50–100), repeats.

## 8. Confounds

List ≥3 ways I could get the H1-predicted result without H1 being true.

- The base model amplifies the same traits as the ones with varying scarcity  → ruled out by training the base model without using scarce rewards / accepted as limitation

- Long answers being rewarded and given as descriptiveness → ruled out by introducing 1 insight per sentence, not counting repetition as a new insight / accepted as limitation

- One model changed more just because of more training  → ruled out by training both models the same distance, while checking how far each drifted / accepted as limitation

- The dataset could be imperfect and could offset the center of the concept → ruled out by nudge-test arrow / accepted as limitation

## 9. Sanity checks

- **Positive control:** I briefly fine-tune the model on overly confident texts, and then measure if the overconfidence dial shifted clearly. If the shift isn’t detectable, then it means the instrument is broken.

- **Negative control:** I measure the untouched model twice, on 2 different seeds. The results must come out about the same. The difference between them is the wobble, which I will take into the account, which fine-tuned models must beat.

- **Manual spot-check:** I will read transcripts to see the changes my dials won’t see.

## 10. MVP, kill criteria, decision points

- **MVP (1–2h version):** Fine-tune on confidence arrow, read the base model, read ~10 transcripts. Only expand, if the arrow passes the nudge test.

- **Kill criteria:** A result of both rich and poor models landing within one wobble of each other on all four dials, while the instrument (positive control) passed falsifies the idea; I write it up as negative and stop.

- **Decision points:** If by hour 7-8 the fine tuning hasn’t moved the dial, I pivot to shipping measurement study as a write-up

## 11. Analysis plan

- **Headline plot/table:** 4 traits on x-axis, shift from base in wobble units on y-axis. 2 points to represent rich and poor for each trait. H1 is true if they separate by more than one wobble, H1 is false if they all stay within one wobble distance.

Stats (or, for small-N sprints: effect size + eyeball, stated as such): No p-values. One model per arm. A shift counts only if it beats the wobble mentioned in section 9. Anything I slice after I see the results is labeled exploratory.

## 12. Lab notebook (running log)

- **Every run:** config, seed, result, one-line interpretation. Include failed, weird, and negative runs. Save raw outputs — you will want to reread transcripts.

| Run | Config / seed | Result | Interpretation |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

## 13. Write-up skeleton

- **Headline claim (one sentence):** ________

- **Key figure:** ________

What I did / what I found / what it means / limitations (from §4 gaps + §6C method-gap + §8 confounds not ruled out) / what I'd do next:
