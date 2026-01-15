# Learnable Parameters

## 1) What “learned parameters” really are (stripped of mystique)

At the lowest level, a learned parameter is simply this:

> A degree of freedom that the optimizer is allowed to tune instead of you hard-coding a decision.

Nothing more.

Every learned parameter answers a question of the form:

* *How strong?*
* *How important?*
* *How much influence?*
* *In which direction?*
* *Compared to what else?*

When you write a constant, a heuristic, or a rule, you are answering that question **by hand**.
When you introduce a parameter, you are saying:

> “I don’t know. Let the data decide.”

That’s the real trade.

---

## 2) The “greater truth”: learning replaces design certainty with optimization

Here is the meta-principle that keeps repeating:

> **Learning is a mechanism for turning uncertainty into structure.**

You use learned parameters when:

* you know *what kind* of structure is needed
* but not *where* it should live
* nor *how strong* it should be

Attention weights, gates, projection matrices, mixture weights, adapters, prompts, LoRA ranks, SE scalars, etc. all fall into this category.

They are not magic.
They are **placeholders for ignorance**.

---

## 3) The key design question you should ask yourself

Whenever you are tempted to add a learned parameter, ask:

> “What decision am I currently making implicitly or heuristically?”

If the answer is:

* fixed rule
* fixed threshold
* fixed weighting
* fixed routing
* fixed importance
* fixed scale
* fixed interaction pattern

then that is a **candidate for learning**.

---

## 4) A concrete mental model: knobs vs switches

### Hard-coded logic = switches

* on / off
* use A or B
* apply or skip
* fixed behavior

### Learned parameters = knobs

* continuous
* differentiable
* adjustable by gradient descent
* softly express preferences

Deep learning systems overwhelmingly prefer **knobs over switches** because:

* gradients flow
* training is stable
* behavior adapts smoothly
* no brittle thresholds

This is why you see:

* attention instead of rules
* gates instead of if/else
* weights instead of heuristics
* scores instead of decisions

---

## 5) When *not* to use learned parameters

This part is just as important, and often ignored.

Do **not** add learnable parameters when:

### 1) The decision is known and invariant

Examples:

* causal masking
* padding masking
* normalization constants
* geometry constraints
* physical laws

If the truth is stable, learning it just adds noise.

---

### 2) The model cannot observe the signal

If the data does not contain information to learn the parameter:

* it will overfit
* collapse
* or become meaningless

Learnable parameters only work if the loss *actually depends* on them.

---

### 3) You are compensating for a bad structure

A common trap:

> “Let’s add attention / gates / parameters to fix performance.”

Often the right fix is:

* better inductive bias
* simpler structure
* better data

Learned parameters should refine structure, not rescue it.

---

## 6) A very useful design ladder (use this)

When designing a model from scratch, think in layers of commitment:

### Level 1: Fixed structure

* topology
* data flow
* invariances
* constraints

These should be **hard-coded**.

---

### Level 2: Learnable strength and importance

* weights
* attention coefficients
* gates
* mixing matrices

This is where most learning lives.

---

### Level 3: Learnable routing or selection

* attention
* soft routing
* MoE
* adapters

Use this only if the task genuinely needs conditional behavior.

---

If you jump straight to Level 3 without solid Level 1 and 2, the model becomes fragile.

---

## 7) Why “learned” keeps showing up everywhere

Because modern deep learning has discovered a very robust pattern:

> Constrain *where* learning happens, but let learning decide *how much*.

Examples you’ve already seen:

* Heads are constrained subspaces, but weights inside them are learned
* Attention structure is fixed, but relevance is learned
* Residuals are fixed, but contribution strength is learned
* Bottlenecks are fixed, but projection matrices are learned
* MHA restricts interaction, Wᵒ re-learns recombination

This pattern repeats because it works.

---

## 8) A practical rule of thumb you can actually use

When designing your own model, ask these three questions in order:

1. **What must be true for *any* valid solution?**
   → Hard-code that.

2. **Where do I know interaction must exist, but not its form?**
   → Add learned parameters.

3. **Where do I not even know *which* interaction matters?**
   → Consider attention or soft routing, but sparingly.

If you can answer all three clearly, your architecture will usually be sane.

---

## 9) One sentence that captures the intuition

Learned parameters are not clever tricks, they are deliberate admissions of uncertainty that let optimization replace brittle design decisions.
