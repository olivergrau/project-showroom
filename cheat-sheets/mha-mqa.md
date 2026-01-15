## 1) Multi-Head Attention vs Single-Head Attention

### Single-Head Attention

**What it is**

* One set of Q, K, V projections
* One attention matrix
* One way to relate tokens

**What it can do**

* Learn a single dominant interaction pattern
* Capture one notion of relevance at a time

**Limit**

* If multiple relationships matter simultaneously, they are forced into one space
* Different interaction types interfere with each other

---

### Multi-Head Attention (MHA)

**What changes**

* Split the embedding into multiple subspaces
* Each head has its own Q, K, V
* Attention is computed independently per head
* Results are concatenated and projected back

**Key idea**
Different heads learn **different interaction patterns in parallel**.

**What it buys you**

* Multiple relational views at the same time
* One head can focus on syntax, another on semantics, another on position or long-range context
* Reduced interference between patterns

**Cost**

* More projections
* More memory traffic
* More compute

**Rule of thumb**
Multi-head attention increases **representational richness**, not conditional computation.

---

## 2) Multi-Query Attention vs Single-Query (standard) Attention

### Standard Attention (single query per head)

**What it does**

* Each head has its own Q, K, and V
* Every head attends independently

**Problem at scale**

* K and V must be stored per head
* During inference, memory bandwidth dominates
* Especially painful for long sequences and autoregressive decoding

---

### Multi-Query Attention (MQA)

**What changes**

* Each head has its own Q
* All heads share the same K and V

**Key idea**
You keep multiple attention perspectives through Q, but **reuse memory-heavy components**.

**What it buys you**

* Much lower memory usage
* Faster decoding
* Better cache locality
* Almost the same quality as full multi-head attention in many tasks

**What you lose**

* Less flexibility in how values are represented per head
* Slight reduction in expressiveness

**Important observation**
Most of the diversity comes from Q, not from duplicating K and V.

---

## 3) Putting them together

| Pattern     | What is duplicated | What is shared | Main benefit                |
| ----------- | ------------------ | -------------- | --------------------------- |
| Single-Head | nothing            | everything     | simplicity                  |
| Multi-Head  | Q, K, V            | nothing        | representational diversity  |
| Multi-Query | Q                  | K, V           | memory and speed efficiency |

---

## 4) Intuition in one sentence each

* **Multi-Head Attention** says: “Let me look at the same sequence in several different ways at once.”
* **Multi-Query Attention** says: “Let me keep those different viewpoints, but stop duplicating expensive memory.”

---

## 5) Why this matters in practice

* Training often prefers **full multi-head attention** for flexibility.
* Inference often prefers **multi-query or grouped-query attention** to reduce memory pressure.
* This is a classic example of replacing raw capacity with **better system-level efficiency**.
