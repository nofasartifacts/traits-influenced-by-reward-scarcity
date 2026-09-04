# Research Design Template

Fill top to bottom **before writing code**. Sections 1–11 are mandatory before the first run; 12–13 you fill as you go.
If the idea is too vague to fill sections 1–5, that's a sign to run a 1–2h *exploration* whose only goal is producing a fillable Section 1 — then come back.

---

## 0. Metadata
- Date / idea origin:
- Timebox (total hours):
- Mode: ☐ exploration ☐ confirmation

## 1. Research question
One sentence. Ends in "?". Must be answerable with "no".

>

**Test:** Is there a possible result that would make you say "the answer is no"? If every outcome "would be interesting," you don't have a question yet — rewrite.

## 2. Why it matters (the delta)
- What belief does the answer update, and whose?
- One line: *If true → ___. If false → ___.*
- Honesty check: how sure are you of the answer already? If >90%, this is a demo, not an experiment. (Demos can still be worth building — just label them.)

## 3. Prior work — hard cap 30–60 min for a sprint
3–5 closest papers/posts. For each:

| Work | What they did | What they found | My delta |
|---|---|---|---|
| | | | |

If your delta is still unclear after the cap, the idea needs reshaping, not more reading.

## 4. Operationalization
Every fuzzy word in Section 1 ("deception", "understands", "capability", "reasoning") gets a concrete, measurable proxy.

| Construct | Proxy / measurement | Gap between proxy and construct |
|---|---|---|
| | | |

The gap column goes verbatim into Limitations later. If you can't fill the proxy column, you can't run the experiment.

## 5. Hypotheses & predictions — written BEFORE any run
- **H1 (mine):** ___ → predicts I'll observe ___
- **H0 (null):** ___ → predicts ___
- **H2 (the boring explanation):** ___ → predicts ___

| Hypothesis | If true, I'll see | If false, I'll see |
|---|---|---|
| H1 | | |
| H2 | | |

## 6. Method & tools
How you'll actually intervene and measure. Chosen on purpose, not by default.

**A. Chosen method/tool**
- Method/tool:
- Why it fits: what does it let me observe / isolate / manipulate that §1 requires?
- Alternatives I rejected:

| Alternative | Why not |
|---|---|
| | |

If you can't name a single rejected alternative, you didn't choose — you defaulted.

**B. Simplest possible method**
The dumbest thing that could in principle answer §1 (often: prompt the model, read the outputs by hand, count).
- Simplest method here:
- Why use it: cheapest, fastest, fewest moving parts — and fewest hidden assumptions, so fewest ways to fool yourself.
- Why it's not enough here (if it isn't):
- **Rule:** anything fancier than the simplest method must be justified in one sentence — what exactly does the upgrade buy? No sentence → use the simple one.

**C. The ideal instrument (unrealistic allowed)**
If no available method truly works — or even if one does — describe the tool that *would* answer §1 perfectly: what it would need to see, isolate, or manipulate, ignoring feasibility entirely.
- Ideal instrument:
- Which of its capabilities my chosen method covers:
- Which it misses → this is the **method-gap**; it goes into Limitations (§13).
- If the overlap is ~zero, the question isn't answerable with this method. Reshape the question or the method before proceeding.

## 7. Design
- **Independent variable(s)** — what I manipulate, and its levels/conditions:
- **Dependent variable(s)** — exact metric, exact computation:
- **Baseline / control condition** — the *strongest boring alternative*, not just "no intervention". (Matched length? Matched tokens? Placebo version of the intervention?)
- **Held constant:** model(s), temperature, seeds, prompt scaffold, decoding params…
- **Sample:** N per condition, which models, which data, how selected:

## 8. Confounds
List ≥3 ways I could get the H1-predicted result *without H1 being true*.

1. ___ → ruled out by ___ / accepted as limitation
2. ___ → …
3. ___ → …

## 9. Sanity checks — run these FIRST
- **Positive control** (must succeed or the harness is broken):
- **Negative control** (must fail or the metric is meaningless):
- Manual spot-check: read N raw transcripts by eye before trusting any aggregate number.

If a sanity check fails: stop, fix, rerun. Do not proceed.

## 10. MVP, kill criteria, decision points
- **MVP (1–2h version):** 1 model, ~10–20 examples, eyeball the outputs. Scale up only if MVP shows signal AND sanity checks pass.
- **Kill criteria:** "A result of ___ falsifies the idea; I write it up as negative and stop."
- **Decision points:** "If by hour ___ I don't have ___, I pivot to ___ / stop."

## 11. Analysis plan — pre-committed
- Headline plot/table: sketch the axes *now*. What does it look like if H1 is true? If false?
- Stats (or, for small-N sprints: effect size + eyeball, stated as such):
- Anything you slice or filter *after* seeing results is **exploratory** — label it as such in the write-up. No silent re-slicing until something looks good.

## 12. Lab notebook (running log)
Every run: config, seed, result, one-line interpretation. Include failed, weird, and negative runs. Save raw outputs — you will want to reread transcripts.

| Run | Config/seed | Result | Interpretation |
|---|---|---|---|
| | | | |

## 13. Write-up skeleton
- Headline claim (one sentence):
- Key figure:
- What I did / what I found / what it means / limitations (from §4 gaps + §6C method-gap + §8 confounds not ruled out) / what I'd do next:

---

## Pre-flight checklist — 60 seconds before starting
- ☐ Question is falsifiable (§1)
- ☐ Fuzzy constructs operationalized, gaps stated (§4)
- ☐ Predictions written before data (§5)
- ☐ Method chosen over ≥1 named alternative; anything beyond the simplest justified in one sentence (§6)
- ☐ Baseline is the strongest boring alternative (§7)
- ☐ ≥3 confounds listed (§8)
- ☐ Positive + negative controls defined (§9)
- ☐ Kill criteria + timebox set (§10)
- ☐ Committed to reporting the runs that didn't work (§12)
