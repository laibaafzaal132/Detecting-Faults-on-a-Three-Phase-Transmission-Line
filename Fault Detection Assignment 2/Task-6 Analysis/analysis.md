# Task 6 – Analysis

All numbers below come from the executed notebook (Tables 1–5 in `results/tables.md`). The test set has 2401 rows: 1305 no-fault and 1096 fault.

## 6.1 Why do logistic regression on Set R and the linear SVM score almost the same, and what does it say about the data?

Both models draw the same kind of decision boundary: a straight line (a hyperplane) in the six-dimensional space of standardised Ia, Ib, Ic, Va, Vb, Vc. They only differ in *how* they pick that line (log-loss versus maximum margin), so if no good line exists, both end up at nearly the same poor one. That is exactly what we see on the test set: logistic regression on Set R scores **72.64 %** and the linear SVM scores **73.64 %**. Both have **0 false alarms** and miss **657** and **633** of the 1096 faults (about 60 % and 58 %). Changing the linear SVM's C from 0.1 to 100 barely moved validation accuracy (0.7258 → 0.7262), and logistic regression's training cost stayed near 0.677 for every usable α, so the optimiser is not the problem.

This tells us the two classes are **not linearly separable** in the raw measurements. The reason is physical: the signals are sinusoids. A single sample of a faulted Ia can be +880, −880 or 0 depending on where in the cycle it was taken (see the Task 2 plot), and a healthy sample can also be positive or negative. A fault changes the *size* of the waveform, not the *sign* of any one sample, and a straight line in (Ia, Ib, …) can only respond to sign and direction. The information is in the magnitudes, which need a non-linear model or squared features.

## 6.2 Signs of the I² and V² weights in Set Q

Learned weights (standardised features, Table 5):

| Weight | Value | Weight | Value |
|---|---|---|---|
| Ia² | **+18.89** | Va² | **−6.03** |
| Ib² | **+4.75** | Vb² | **−6.46** |
| Ic² | **+13.96** | Vc² | **−6.14** |

**Current-squared weights are positive.** When a fault occurs, a low-impedance path to ground or between phases opens up and the current surges, often by an order of magnitude (Ia reaches about ±880 in the Task 2 plot, against about ±60 normally). A large I² therefore pushes the fault score up, so the weight must be positive. Squaring removes the sign of the sample, so the model sees the amplitude regardless of where in the cycle the sample was taken.

**Voltage-squared weights are negative.** The same short circuit pulls the voltage at the relay towards zero (the voltage collapses across the fault). A large V² therefore means the line is healthy, and a small V² points to a fault, so the weight must be negative. All three V² weights are close to each other (−6.0 to −6.5), so the model treats a sag on any phase as similar evidence. The current weights differ more (Ia² 18.9, Ic² 14.0, Ib² 4.8). I would not read much into those differences from the weights alone: they depend on how large the current swings are on each phase in this dataset and on the standardisation scale of each squared feature. What matters here is the sign.

In short: *high current and low voltage means fault*, which is the textbook signature of a short circuit.

## 6.3 Support vectors: best linear SVM versus best RBF SVM

The best linear SVM (C = 100) uses **5835 support vectors**, which is **81 %** of the 7200 training rows. The best RBF SVM (C = 100, γ = 1) uses only **266**, or **3.7 %**.

A very large number of support vectors means that most training points sit on or inside the margin or on the wrong side of it. The model cannot find a clean gap between the classes, so nearly every point ends up "mattering" for the boundary. That is a sign that the model class is too simple for the data (underfitting), which agrees with the 73 % accuracy. It also makes prediction slower, because every new sample is compared with every support vector. The RBF SVM's small set means a few points near the true boundary are enough to describe it, which is what we expect when the model fits the structure of the data.

## 6.4 What happens to accuracy and support vectors as C increases?

From the validation table:

| Model | C = 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|
| Linear – accuracy | 0.7258 | 0.7258 | 0.7258 | 0.7262 |
| Linear – SVs | 5849 | 5843 | 5835 | 5835 |
| RBF γ=0.1 – accuracy | 0.9500 | 0.9750 | 0.9921 | 0.9954 |
| RBF γ=0.1 – SVs | 1347 | 723 | 381 | 174 |
| RBF γ=1 – accuracy | 0.9804 | 0.9917 | 0.9962 | 0.9967 |
| RBF γ=1 – SVs | 848 | 482 | 342 | 266 |

For the RBF SVM, **accuracy rises and the number of support vectors falls as C grows.** For the linear SVM nothing really changes.

C is the price the SVM pays for each training point that violates the margin. With a small C the optimiser prefers a wide margin and a smooth boundary and tolerates many violations. Many points then lie inside the margin, and every one of them is a support vector, so we get many support vectors and a boundary that is too smooth to follow the data (lower accuracy). With a large C, violations are expensive, the margin narrows, and the boundary bends to put points on the correct side. Fewer points remain inside the margin, so fewer support vectors are needed and accuracy improves. The linear SVM cannot bend, so no value of C can remove its violations: its support-vector count stays near 5840 and accuracy stays near 73 %.

One caution: going from C = 10 to C = 100 at γ = 1 improved validation accuracy by 0.0005, which is a single row out of 2400, so it is within noise. A very large C can overfit, so the extra gain should not be over-interpreted.

## 6.5 Is the test accuracy optimistic or pessimistic for a real relay?

**Optimistic.** The rows are consecutive samples of smooth waveforms, so a test sample has near-identical neighbours (one sample earlier and later) in the training set. A flexible model such as the RBF SVM can score highly on the test rows simply because it has effectively seen almost the same sample, which is leakage through time-adjacency rather than evidence that it understands new faults. This probably flatters the models with the most capacity most, and it fits what we see: training, validation and test accuracy for logistic regression Set Q are almost identical (97.9 / 97.8 / 97.9 %), so the random split cannot reveal a generalisation gap.

A real relay has to handle events it has never seen: a different fault location, fault resistance, inception angle, fault type, and noise. The hardest samples are the first few milliseconds after a fault begins or clears (the Task 2 plot shows high-frequency transients there), and these are only a small share of a random test set. A fairer test would split by contiguous time blocks or by whole fault events (for example, leave-one-event-out), and I would expect lower accuracy on it, with the biggest drop for the RBF SVM.

## 6.6 Which model would I deploy?

I would deploy the **RBF SVM (C = 100, γ = 1)**, judged on errors rather than accuracy alone:

| Model | Missed faults (of 1096) | Miss rate | False alarms (of 1305) |
|---|---|---|---|
| LR, Set R | 657 | 59.9 % | 0 |
| Linear SVM | 633 | 57.8 % | 0 |
| LR, Set Q | 45 | 4.1 % | 6 |
| **RBF SVM** | **6** | **0.55 %** | **4** |

A missed fault means the breaker does not trip while the line is faulted, which risks equipment damage and a wider outage. A false alarm causes an unnecessary trip, which is costly but recoverable. The two linear models have zero false alarms but miss almost 60 % of faults, so they are unusable as protection however "safe" they look. The RBF SVM has both the fewest missed faults and the fewest false alarms; logistic regression on Set Q is next best and is simpler, transparent and cheap to compute, but it misses about 7× more faults.

Caveats before real deployment: (1) the 0.55 % miss rate is optimistic (6.5), so I would re-measure it with a time-block or event-based split; (2) I would shift the SVM decision threshold slightly towards "fault" to trade a few more false alarms for fewer misses; (3) I would require a short run of consecutive fault samples before tripping, which removes isolated false alarms at the cost of a little delay; and (4) I would keep the logistic regression Set Q model as a cheap, interpretable cross-check.
