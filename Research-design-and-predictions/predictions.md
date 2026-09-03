# Predictions

Predictions are taken from chapter 5 of the research design document, and pasted in here. The reason for it is that research design doc will change (chapters 12 and beyond, where I will fill info in while doing the research), and I want predictions to stay posted before the research is done.

- **The wobble:** I measure the untouched model twice, on 2 different seeds. The results must come out about the same. The difference between them is the wobble, which I will take into the account, which fine-tuned models must beat.

- **Analysis plan:** 4 traits on x-axis, shift from base in wobble units on y-axis. 2 points to represent rich and poor for each trait. H1 is true if they separate by more than one wobble, H1 is false if they all stay within one wobble distance. No p-values. One model per arm. A shift counts only if it beats the wobble. Anything I slice after I see the results is labeled exploratory.

| Hypothesis | If true, I'll see | If false, I'll see |
| --- | --- | --- |
| H1: Reward scarcity changes where the persona’s traits move | Deception vs honesty: Rich model will become more deceptive.<br>Confidence vs hedging: Rich model will become more confident, since before that it was rewarded for almost every answer.<br>Descriptiveness vs vagueness: Rich model will become more vague, because it did not have to perfect its speech and ability to reason properly to get rewards before.<br>Agreeableness vs pushback: Rich model will start pushing back more, cause it is more confident.<br>Whereas the poor model will not move much from the measured traits. | Rich and poor are within one wobble of a difference. |
| H0: Every model develops equally, no meaningful difference between base model and those with varying reward scarcity. | Difference between the base and tested are within one wobble | Beyond one wobble of a difference |
| H2: Traits do get amplified, but scarcity is irrelevant. | The rich and poor differ from the base (above one wobble of difference), but rich and poor are within one wobble of a difference between each other. | Either nothing, or H1 |
