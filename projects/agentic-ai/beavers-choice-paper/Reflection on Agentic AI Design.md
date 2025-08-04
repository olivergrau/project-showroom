# Reflection on Agentic AI Design: A Software Engineer’s Perspective

As a seasoned software engineer with over 20 years of experience building systems using Domain-Driven Design (DDD), clean architecture, and enterprise-grade software principles, I’ve learned to value clarity, determinism, and maintainability in all layers of a system. 

After implementing and dissecting the Udacity Agentic AI project for Beaver’s Choice Paper Company, I feel compelled to share a few hard-earned reflections on the current state of LLM-based agent design. While the academic premise of agentic AI is intriguing, its real-world applicability—especially for deterministic, transactional domains—falls short.

---

## 🧱 1. The Illusion of Delegation

Agentic AI systems give the impression that “intelligent agents” will take over decision-making and action execution. But in practice, the developer ends up doing all the actual work:

- Precomputing values like stock levels and cash balances
- Writing the tool logic that performs actual transactions
- Guarding against hallucinations and invalid reasoning
- Verifying the correctness of every output

What remains for the agent? A verbose justification for decisions already made in code.

---

## 🤖 2. LLMs Are Not Deterministic Executors

In real software systems, especially those managing inventory, finances, or orders, we need guarantees—not guesses. The agent framework asks the LLM to reason over stock levels or cash balances, but it frequently produces factually incorrect conclusions (e.g. “659 is sufficient for 1000”).

This violates fundamental expectations in software architecture:
- Deterministic logic
- Verifiable state transitions
- Predictable behavior

LLMs can't be trusted with logic that needs to be correct every time. They narrate workflows, not execute them.

---

## 🧠 3. Tools Are Still the Real Actors

Despite the language around "agents acting," all meaningful work is still done via code tools. Whether it's creating a transaction, checking inventory, or fetching prices, the logic resides in functions written by humans.

The agent may “decide” to perform an action, but it cannot execute it—it merely *suggests* the call. This breaks the separation of concerns and blurs accountability: Who made the decision? Who validated it? Who ran it?

---

## 🔧 4. Misalignment with Proven Engineering Practices

Agentic workflows often violate basic architectural principles:
- No clear boundaries between control flow, logic, and side effects
- No unit tests or validation for what the LLM decides
- No composability or reuse—each “agent” becomes a black box

This is in direct contrast to the systems I’ve built using DDD, where each domain concept is explicit, traceable, and testable.

---

## ⚖️ Exploring Two Approaches for a Deterministic Domain

During the project, I considered two architectural strategies for implementing the Beaver’s Choice quote processing system—both centered around how to best apply (or not apply) Agentic AI principles in a deterministic business domain.

### **Option 1: Full Agent-Based Architecture (Implemented)**

In this approach, I embraced the Agentic AI paradigm as defined in the project requirements:

- Created multiple domain-specific agents (InventoryAgent, QuoteAgent, OrderAgent, etc.)
- Introduced an OrchestratorAgent to coordinate their interactions
- Used tool interfaces to encapsulate specific operations (get inventory, create transactions, etc.)

**Result:**  
This provided a valuable hands-on exploration of agent-based design. However, the downside was clear: I had to continuously guard against LLM misjudgments, hallucinated reasoning, and incomplete execution. Most of the meaningful logic still had to be replicated and validated outside the agents.

### **Option 2: Logic-in-Tools Only (Not Realized)**

This alternative would enforce a strict rule:

> *All critical business logic is implemented inside deterministic tools.  
> The LLM merely wraps those tools to produce explanations or trigger flows.*

While more stable, this approach quickly raises the question:  
**If all important logic is in tools, why use LLMs at all?**

It becomes a system where a deterministic backend is controlled by an uncertain frontend, offering little practical benefit over modern software architectures—particularly Domain-Driven Design (DDD), where domain logic is explicit, testable, and versionable.

---

**Conclusion:**  
Option 2 offers no unique value over DDD. It reintroduces complexity without introducing capability. In deterministic domains, wrapping LLMs around imperative tools does not constitute innovation—it’s overhead.


## 💡 5. Where LLMs *Do* Add Value

In my view, LLMs shine when used *around* the system—not *inside* it:
- Summarizing logs, reports, or state transitions
- Generating explanations for quote decisions
- Assisting with natural language queries or debugging

They are powerful as **augmenters**, not controllers.

---

## ✅ My Guiding Principle Going Forward

> If something must be correct, it should be in code.

LLMs are not a replacement for architecture. They are an interface.  
In serious systems, I will continue to rely on robust logic, clear constraints, and explicit rules—while letting LLMs enhance interpretability and communication.

---

## 📌 Final Thought

The Agentic AI movement is an exciting frontier, but we must separate **the hype from the usable**. In their current state, agentic architectures built around LLMs are not ready to replace well-engineered workflows in deterministic domains.

I’m glad I explored it. But I’ll be even happier maintaining clean, reliable code that does what it says—and says what it does.

---

## 🎓 Final Note: A Valuable Learning Experience

Despite the practical limitations and architectural friction, completing the Beaver’s Choice Paper Company project was a genuinely valuable experience. It forced me to confront the realities of Agentic AI—both its promise and its pitfalls.

I now have a much deeper understanding of:
- Where LLM-based agents break down
- How fragile their reasoning can be in structured domains
- What it takes to bridge deterministic code with probabilistic decisions
- How to design fallback-safe, testable interfaces between language and logic

This wasn’t just a project. It was a field test for the limits of current Agentic AI approaches—and for that, I’m grateful. I walk away with sharper instincts, clearer architectural judgment, and firsthand knowledge of what Agentic AI *can* and *cannot* do today.

