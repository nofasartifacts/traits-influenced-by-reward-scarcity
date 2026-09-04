# Research Design for: Does abundance of rewards shift traits differently than a scarce one in LLM?

## Executive summary

Headline claim:

Imitation fine-tuning moves the confidence trait of Qwen3-1.7B by 7-9 units (wobbles), while 50 GRPO steps at 89% or 6% reward move it less than one. Tested by an internal ruler/arrow in the thinking block, validated the arrow measuring the trait by controls like: affinity (which words it promotes and suppresses at different layers), fingerprints (the 2 sides separate without pairing from layer 12), nudge test (nudging towards one concept, and towards the opposite), the wobble (which is the unit), and the positive control (which gives the 7-9 wobbles mentioned earlier)

Key figure:

Positive control +8.6 wobbles, rich −0.4, poor −0.4, noise band ±1

What I wanted to do:

There’s a reasonable believe that LLM’s persona is shaped during the pre-training. I wanted to find out if an LLM’s persona could also be shaped during the post-training/fine-tuning. To test it I had came up with this method: does a different reward scarcity during post-training/fine-tuning affect some specific traits differently.

Why I wanted to do:

Because changing the pre-training is hard - all you do is feed information into a future-LLM. It is expensive, and it is a lot. It would be very difficult and expensive to try and blabber with the information you feed it. But post-training is more hands-on, more customizable. If you could adjust the way / find a new way you post-train/fine-tune the model, the opportunities to properly adjust, or align a misaligned model would increase.

What research looks like in theory:

I find traits I want to test, then train the model on math problems: giving one version of a model easier problems, that it can solve in ~90% of cases (rich pool), and giving another version of the same model harder problems, that it can solve in ~5% of cases (poor pool). Then I test the 3 versions of this model (rich, poor, base) on non-math questions. And then I compare the three, and see if anything’s changed.

The setup:

Rented RTX 4090, Qwen3-1.7B, bf16, temperature 0.6, top_p 0.95, top_k 20, cap 6,000 for language tests, and 3,000 cap for fine-tuning on math problems.

What I did:

Chose the confidence vs hedging trait, collected 4 thinking blocks from a local model of Qwen3-1.7B as examples for a smarter LLM to imitate, read them, found the marker words.

Constructed 100 questions with 100 pairs of answers, where I had 100 hedging answers that would include 25 to 39 markers for 1,000 words, and 100 confident answers with 0 markers for 1,000 words. With length ratio of confident to hedging answers kept between 0.90 to 0.98, and having both traits having the same final answer. Then 100 questions only for a test. And then 100 more questions with 100 pairs of answers.

Built the arrow from 80 pairs, tested on 20. Looked at it through the output matrix and fingerprints to determine whether the arrow is correctly determining the concepts.

Pushed the arrow along the confidence concept on 2 layers (12 and 16), and read transcripts to make sure it is affecting the model.

Tested the base model on questions dataset twice (2 different seeds) to determine and define the wobble unit.

Performed a positive control - fine-tuned on 50 confident thinking blocks, and compared it to the base seeds, measured the difference in wobble units.

Sorted 2 pools of math problems by giving it to the base model, where solved easy problems would go into the rich pool, and barely solved difficult problems would go into the poor pool. After that I trained two arms for 50 steps, and compared the results of each arm to each other and to base model’s 2 seeds. All was measured in wobbles.

What I found:

Imitation fine-tuning moves the confidence trait, but also makes it loop more. Hedging seems to override correct facts in Qwen3-1.7B. 50 steps of GRPO was NOT enough to shift the confidence train further than the mean of wobble. And a few numbers and stats: arrow accuracy 1.00, effect 16; affinity words; nudge 51 to 37; wobble 0.99; positive control +8.6 and +7.1 clean, hedge words 24 to 5; pools 89% and 6%; arms −0.4 and −0.4, clean −0.1 and −0.5; rich drift in steps 51–100.

What it means:

The trait is readable, and very movable in post-training, when it is trained on the trait. However, at 50 steps of GRPO, while training on math questions, regardless of it is rich or poor pool, the difference is not visible - it stays within the wobble.

Limitations:

Tested only 1 trait, only on 1 relatively small model, with just 1 seed per arm, and only 2 seeds of base to determine the proper wobble boundaries. 50 steps instead of planned 100, but even that seems like I’d need more than 100 steps. Tiny weight distance. Quite restrictive cap on tokens.

What I’d do next:

Re-test the trait with more steps: 100, 200, 300 to see how it changes, with a bigger cap on tokens. Then test the remaining traits. Do positive controls on different models, bigger ones, to see if this concept generalizes throughout models. And do several seeds per arm.

Repo:

https://github.com/nofasartifacts/traits-influenced-by-reward-scarcity

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

Every run: config, seed, result, one-line interpretation. Including failed, weird, and negative runs.

| Run / date | Config / seed | Result | Interpretation |
| --- | --- | --- | --- |
| Sept 2, base observation on local model | Qwen3-1.7B locally in Ollama, no cap. | Four think blocks: 3,454 / 13,667 / 2,532 / 2,039 words. On question 2 it got stuck in a loop for a bit, 840 s. Backtrack markers: 42, 12, 19, 36 per 1,000. The full list of the backtrack markers is gonna be in chapter 14. | To get 4 examples of reasoning, so that I could feed to an LLM to imitate to create a dataset |
| Sept 3 | In scope of confidence vs hedging only. 100 questions and pairs of answers (100 confident answers and 100 hedging answers)) written by an LLM, by imitating Qwen’s thinking style. | 100 questions with 100 same answers of each side, same length within 10%, confident is an edit of hedging with markers removed, contractions kept. And an additional 100 new questions | 100 pairs to train an arrow. 100 questions to test the base model 2 times to determine the wobble (the difference between 1st and 2nd seeds); to perform positive control after fine-tuning on overconfident text; and to test rich and poor model after GRPO |
| Sept 3, base model on a rented GPU | RTX 4090, bf16, temperature 0.6, top_p 0.95, top_k 20, cap 2,000. | Think block present, correct answer, 83 tokens/s, 4.4 GB peak. | A quick check before running the dataset of questions |
| Sept 3 | build_arrow: seed 0, 80/20 | The accuracy is 1.00 at all layers, held-out effect peak 16.4 at layer 8, layer 0 separates too. | The arrow separates the pairs, word-level component exists (cause separation is done from layer 0). What isn’t certain, is if it’s only word-level separation, or if it detected at the level of concept as well |
| Sept 3 | arrow_affinity: layers 0–28 | At layers 12–24 it promotes "consequently, thus, 必将" and suppresses "maybe, suppose, reconsider, clarify, myself"; at layer 0, digits against "me". | The model seems to detect separation at concept level, since it (the arrow) promoted and suppressed words beyond the marker list.<br>I theorize that digits were promoted in layer 0 because in the prompt for dataset creation to minimize the gap of words it was suggested to add more calculations. |
| Sept 3 | plot_fingerprints: layers 0–20 | gap over spread rises from 0.5 at layer 0 to 2.4 at layer 16 | The rows separate without pairing from layer 12 on |
| Sept 3 | Nudge test | Layer 16, markers per 1,000: −4 48.6, −2 51.3, 0 43.8, +2 36.7, +4 36.8. Finished at cap 3,000: 0–3 of 5. Dose window ±2; +4 leaves the distribution. | I read 3 versions of answers for each question, 2 question in total. They were read on 0, -2, and +2.<br>On 0 (base model) the model was backtracking quite a bit (as usual), but consistent with what was demonstrated before.<br>On -2 (pushed towards hedging) the model was backtracking quite a bit, but also getting caught up on the information it already checked, and getting confused more than usual. Making incorrect calculations/assumptions and believing them<br>On +2 (pushed towards confidence) the model was backtracking less, which led to the LLM to get to the next components of questions faster.<br>The cap of 3,000 is not enough, will be upped to 6,000. |
| Sept 3, base reads | the plain model took the exam twice, seed 1 and seed 2, cap 6,000, dial read at layer 16. | dial −39.86 and −38.88, wobble 0.99. Hedge words 33.1 and 30.0 per 1,000, wobble 3.2. Finished 91 and 86 of 100. Loops 183 and 145. | The wobble is the difference between two seeds within the same unchanged model. The wobble is about 1/8 of the distance between confident and hedging centers. Also, a sidenote: the base model loops a lot, and is already far on the hedging side. |
| Sept 4, fine-tuning by imitation | 50 confident think blocks (rows 1–50 of the second (new) set). Questions 1-50 became the training set, and 51-100 were the unseen questions. LoRA rank 16, 3 passes, learning rate 0.0001. | 23 seconds. Loss from 1.83 to 1.25. | It seems like the model learned the confident style, and didn’t just memorize the text markers |
| Sept 4, read of the fine-tuning | The fine-tuned model took the exam once, seed 1, cap 6,000. | On the unseen half (questions 51–100), the dial moved from −40 to −31, a shift of +9, which is 11.8 wobbles. Hedge words 24 to 5. On the trained half, +7.9. Finished 53 of 100, loops 248. | Less finished thinking blocks, more loops. But the shift persists on the questions it hasn’t seen, which is why I assume it is a trait-level change, not just recitation. |
| Sept 4, read of finished-only | same reads, unseen half, only the 25 questions where every read finished. | dial +8.1, which is 7.5 wobbles. Loops 13, 7, 11. Fine-tuned blocks 551 words, base 1,300. | I wanted to check if loops carried the results of the previous check - they did not. |
| Sept 4, pools of questions on base model | 150 easy and 300 hard problems, one base attempt each, cap 3,000, a win must finish | Easy 110 of 150, 73%; hard 52 of 300, 17%; every hard attempt that finished was a win; rich pool 110, poor pool 120 (36 AIME, 49 level 5, 35 level 4); longest wins 2,920 and 2,988 tokens | Pools were sorted with a cap of 3,000 tokens. On hard problems model pretty much just never answered, when it didn’t know. It either answered correctly, or never answered at all (loops, without ever committing before the cap of 3,000 tokens hits it) |
| Sept 4, start of the research question, fine-tuning by reward training, rich pool | rich pool from easy school math, poor pool from hard competition math; one base attempt per problem, cap 3,000. Cut answers (longer than 3,000) are removed from learning. Rich arm ran 100 steps, read at step 50. Poor arm ran 50 steps (cut due to lack of time). LoRA, same settings both arms, cap 3,000 in training. | Steps 1–50: reward 0.89, so the rich pool's rate is 89%; 36 of 50 steps with no signal; mean length 1,774 tokens; 11% cut; distance from base 0.0012. Steps 51–100: reward 0.72, length 2,159, 27% cut, distance 0.0154. | Since long answers were cut instead of punished, it seems like the model learned to aim the word count towards the higher number, which may have triggered it to have been looping more. |
| Sept 4, poor pool | Poor pool, same settings as the rich one, 50 steps | Mean reward over the 50 steps: ~ 0.06, so the measured solve rate of the poor pool is 6%, against the 5% target. 43 of 50 steps gave no learning signal, all 8 answers cut or all 8 wrong. 94% of all answers were cut at the cap. Distance from base at the end: 0.0001. | The model learned only on 7 steps out of 50, and weights barely moved. If the design is correct, it will be just about the same as the base model - within 1 wobble. |
| Sept 4, reads of the arms | Rich at step 50 and poor at step 50 took the exam, seed 1, cap 6,000 | Dial −39.72 and −39.76, base −39.86 and −38.88; hedge words 33.4 and 32.3, base 33.1 and 30.0; finished 85 and 88, base 91 and 86; loops 202 and 169, base 183 and 145. | Both arms are not just within the wobble, they are in-between the two seeds. 50 steps was not enough to move shift traits cleanly out of the base boundaries. |
| Sept 4, comparison | Shift from the base average divided by the wobble | Positive control +8.6 wobbles, rich −0.4, poor −0.4; rich and poor 0.04 apart; per question +0.14 ± 3.04 and +0.11 ± 3.61 | H0 at this 50 steps is confirmed. Section 10 kill criteria is technically met. However the positive control showed significant difference, so further testing with more steps should be what is tested next. Weight distance was 0.0012 and 0.0001 for rich and poor arms respectively. |
| Sept 4, finished-only comparison | The same five reads, only the 38 questions that finished in every read. | Dial +7.1, −0.1, −0.5 wobbles; hedge words −9.6, +1.3, +0.7; words −15.9, +2.8, −0.1; loops 33, 31, 27, 48, 39 | Interpretation holds, rich arm loops more, and thinks a bit longer |

## 13. Appendix

Backtrack markers for confidence vs hedging, they are taken from the original 4 thinking blocks by local Qwen3-1.7B model:

Wait (141), Maybe (110), Alternatively (53), Let me think (30), But that (25), But I need to (16), But maybe (13), Let me check (12), Perhaps (10), Let me try (7), Actually (6), I need to check (6), I'm not sure (3), Hmm (3), Hold on (1), Let me go back (1).
