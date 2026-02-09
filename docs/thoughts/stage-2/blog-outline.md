# Blog Outline: Building the Harness, Not the Demo — A Two-Bot Interview System That Practices What the Essay Preaches

## Meta

- **Target audience**: Engineering leaders and hiring managers who think in terms of the "agentic harness" — orchestration, observability, reliability
- **Thesis**: Stage 2 is a small project that demonstrates big principles. It is not about making two bots talk. It is about building the *system around* two bots talking — the orchestration, the observability, the prompt management, the separation of concerns — and showing that the author thinks like a harness engineer, not a demo builder.
- **Tone**: Technical but opinionated. Same register as the agentic harness essay. Show the work, explain the *why*, connect every decision back to production thinking.
- **Length**: ~3,000–4,000 words (substantial enough to demonstrate depth, short enough to respect the reader's time)

---

## Title Options

1. "Building the Harness, Not the Demo: What a Two-Bot Interview Taught Me About Production AI"
2. "The Model Is Not the Product: Engineering a Multi-Agent System That's Observable, Configurable, and Boring in All the Right Ways"
3. "Two Bots Walk Into a Graph: Orchestration, Observability, and the Engineering That Actually Matters"

---

## Structure

### I. Opening: The Demo Trap (300 words)

**Hook**: Reference the agentic harness essay's core insight — "the demo was the model, the six months was the harness." Flip it: what if you built the harness *first*, even for a small project?

**Setup**: Introduce Stage 2 as a deliberate exercise in harness thinking. Two AI personas interview each other. The conversation is entertaining. But the interesting part isn't what the bots say — it's the system that makes them say it reliably, observably, and configurably.

**Thesis statement**: "I built a small multi-agent system and treated it like a production system from day one. Here's what that looks like — and what it taught me about the three pillars."

**Connection to essay**: Directly name the three pillars (orchestration, observability, reliability) and preview how Stage 2 addresses each.

---

### II. What Stage 2 Does (400 words)

Brief, concrete description of the system. Keep it tight — this is context, not the point.

- Two AI personas (initiator + responder) take turns on a user-provided topic
- LangGraph conditional loop, Mistral AI for generation, Langfuse for prompt management + tracing
- Four persona presets with natural tension built in (Reporter vs Politician, Bartender vs Patron, etc.)
- CLI with streaming output — each response prints as it arrives

**Include**: The state diagram (mermaid) showing the initiator/responder loop with conditional edges. This is the first visual — show the reader the *shape* of the system before diving into decisions.

```
Diagram: State diagram — initiator ↔ responder conditional loop
(Use the stateDiagram-v2 from the README)
```

**Transition**: "The graph is simple. Two nodes, two conditional edges, a turn counter. The interesting engineering is in everything around it."

---

### III. Pillar 1 — Deep Architecture: Orchestration Decisions That Compound (800 words)

This is the longest section. Map every architectural decision back to the essay's first pillar.

#### A. The Node Factory Pattern

- `_build_node_fn(role, prompt_name)` returns a closure for either role
- Eliminates duplication — initiator and responder differ only in persona and turn counter
- **Connection to essay**: "This is microservices architecture applied to cognitive work." The node factory is service decomposition — each node is a specialized worker with its own persona, its own prompt, its own responsibilities. The factory ensures they share structure but not state.

#### B. State Design: InputState vs InterviewState

- `InputState` is the public API — what the caller provides (preset, names, max_turns)
- `InterviewState` extends it with internal counters (`initiator_turns`, `responder_turns`) using `operator.add` reducers
- This is the LangGraph equivalent of separating your API contract from your internal state
- **Connection to essay**: State management in agentic systems is a distributed systems problem. The split prevents callers from accidentally initializing internal counters to wrong values.

#### C. Message Rewriting — The Unglamorous Problem

- Mistral requires strict user/assistant alternation
- Both bots produce `AIMessage`s, so each node rewrites the other bot's messages to `HumanMessage`s
- This is a *real-world constraint* that no demo would surface — it only appears when you wire a second agent into the loop
- **Connection to essay**: "Instead of deterministic functions, you are orchestrating probabilistic reasoning engines. That changes everything." Message rewriting is a concrete example of what "that changes everything" looks like in practice. The model API has opinions about conversation structure, and your harness must accommodate them.

#### D. Conditional Edges and Turn Management

- After each node, a conditional edge checks the *other* bot's turn count
- The graph ends when either bot's partner has no turns remaining
- Turn counters use `operator.add` reducers — each node returns `+1`, and LangGraph handles accumulation
- **Connection to essay**: This is the equivalent of the retry/circuit-breaker logic the essay describes, but for conversation flow control. The conditional edges are the orchestrator "managing the flow of data between agents."

```
Diagram: Flowchart showing the node factory producing both nodes from a single function
— inputs (role, prompt_name) → factory → closure with 5-step pipeline
— annotate each step: fetch prompt, compile persona, rewrite history, invoke LLM, return state
```

---

### IV. Pillar 2 — Observability: Seeing Inside the Conversation (700 words)

Map the Langfuse integration back to the essay's second pillar.

#### A. Automatic Instrumentation via CallbackHandler

- Single `CallbackHandler` passed through `config["callbacks"]`
- LangGraph automatically instruments every node invocation — no manual span creation
- Each trace captures: system prompt, full message history, response, token counts, latency, model version
- **Connection to essay**: "Trace-level visibility into every LLM call: prompt in, completion out, latency, token count, model version." Stage 2 implements exactly this, out of the box, with zero manual instrumentation code.

#### B. Session Grouping

- `main.py` generates a unique `langfuse_session_id` per run
- All 6 traces (3 turns x 2 bots) group under one session in the dashboard
- You can view a full Reporter-vs-Politician transcript as one unit
- **Connection to essay**: This is "agent-level operational metrics" in miniature — you can see success rates, latency per turn, cost per interview, all scoped to a single session.

#### C. The Prompt-to-Trace Link

- `langfuse_prompt` metadata on each `ChatPromptTemplate` ties every trace to the exact prompt version
- If the Politician starts giving straight answers, you check the trace, see it was prompt v4, diff against v3
- **Connection to essay**: "Drift and regression detection that alerts you when output quality changes." The prompt-to-trace link is the mechanism that makes regression detection possible.

```
Diagram: Sequence diagram showing the full trace flow
— User → CLI → Graph → Langfuse Prompts → Mistral → Langfuse Traces
— annotate: session ID grouping, prompt version linking, async flush
(Use the sequence diagram from the README)
```

**Key paragraph**: "The essay warns that 'without purpose-built observability, you are flying blind in a system that is designed to be non-deterministic.' Stage 2 is non-deterministic by design — two bots riffing on an open-ended topic. But every run is fully traced, every prompt version is linked, and every conversation is grouped into a reviewable session. I'm not flying blind. I built the instruments before I took off."

---

### V. Pillar 2.5 — Prompt Management as Infrastructure (500 words)

This bridges observability and reliability. It deserves its own section because it's one of Stage 2's strongest differentiators.

#### A. Prompts Outside Code

- System prompts live in Langfuse as `interview/initiator` and `interview/responder`
- Persona variables (`persona_name`, `persona_description`, `persona_behavior`, `other_persona`) are compiled at runtime
- Editing a prompt is a dashboard operation, not a code change
- **Connection to essay**: "The engineer's primary artifact is no longer a function that computes a result. It is a harness that orchestrates, validates, and monitors something else computing the result." Prompts are the new code. Managing them outside the codebase is the new deployment pipeline.

#### B. Model Config Outside Code

- `model` name and `temperature` live in the Langfuse prompt's `config` field
- Switching from `mistral-small-latest` to another model is a dashboard change
- **Connection to essay**: "The harness is model-agnostic by design. It treats the foundation model as a swappable component." Stage 2's architecture makes the model literally swappable from a dashboard.

#### C. Version History and Rollback

- Every prompt edit is versioned in Langfuse
- You can diff current vs previous, roll back in seconds
- Combined with trace linking, you get a full audit trail: "this conversation used prompt v3, which had this exact text"
- **Connection to essay**: "Audit trails powered by the observability layer." The prompt version history + trace linking is exactly this.

```
Diagram: Flowchart showing prompt management flow
— Langfuse Dashboard (edit/version) → Runtime (fetch/compile) → Trace (link back)
(Use the prompt management diagram from the README)
```

---

### VI. Pillar 3 — Reliability: Where Stage 2 Stops and What Comes Next (400 words)

Be honest about what Stage 2 *doesn't* do — but frame it as intentional scoping, not a gap.

#### A. What Stage 2 Has

- Deterministic flow control (conditional edges guarantee termination)
- Input validation via Pydantic (`InputState` schema)
- Graceful error handling in the CLI
- The *foundation* for the data flywheel (traces exist, prompt versions are linked)

#### B. What Stage 2 Doesn't Have (Yet)

- No reviewer/validator agent checking output quality
- No confidence scoring or escalation paths
- No automated evaluation harness
- **Connection to essay**: "The third pillar separates demos from production systems." Stage 2 is honest about being pre-reliability. But it's *ready* for reliability — the observability infrastructure is in place, the traces exist, the prompt versions are linked. Adding a validator agent that scores conversation quality against a rubric is the natural next step, and the harness is ready for it.

#### C. The Flywheel Is Primed

- Traces capture every conversation
- Prompt versions are linked to outputs
- A future evaluator could score conversations, and failures could feed back into better prompts
- **Connection to essay**: "Observability captures traces of every agent interaction. Failed outputs become evaluation data." The flywheel isn't spinning yet, but the axle is installed.

---

### VII. The Engineering Beneath the Entertainment (300 words)

#### A. Separation of Concerns

- `config.py`: environment and secrets
- `personas.py`: data-only persona definitions
- `graph.py`: orchestration logic
- `main.py`: CLI and streaming
- Adding a new persona is a data-only change — no graph logic, no node code, no prompt templates change

```
Diagram: Mindmap of the file structure
(Use the mindmap from the README)
```

#### B. What This Proves

- You can build a multi-agent system with clean architecture
- Observability is not an afterthought — it's baked in from the first commit
- Prompt management is infrastructure, not string concatenation
- The model is the least interesting part of the system

---

### VIII. Closing: The Harness Is the Skill (200 words)

**Return to the essay's thesis**: "The teams that won were not the ones with the best models. They were the ones with the best harnesses."

**Personal statement**: "Stage 2 is a small project. Two bots having a conversation. But the engineering behind it — the orchestration, the observability, the prompt management, the state design, the message rewriting, the session grouping — is the same engineering that scales to production multi-agent systems in law, healthcare, and finance. The patterns are identical. The domain is different. The harness is the skill."

**Final line**: Something punchy that echoes the essay's closing metaphor about models being hardware and harnesses being the OS. Maybe: "The bots are entertaining. The harness is the resume."

---

## Diagrams Summary

| # | Type | What It Shows | Source |
|---|------|--------------|--------|
| 1 | State diagram | Initiator ↔ responder conditional loop with internal pipelines | README |
| 2 | Flowchart | Node factory producing both nodes from a single function | New — show the factory pattern visually |
| 3 | Sequence diagram | Full trace flow: user → CLI → graph → Langfuse prompts → Mistral → Langfuse traces | README |
| 4 | Flowchart | Prompt management lifecycle: dashboard → runtime → trace → dashboard | README |
| 5 | Mindmap | File structure and separation of concerns | README |

---

## Key Quotes to Thread Throughout (from the Agentic Harness Essay)

Use these as section epigraphs or inline references to create a continuous dialogue with the essay:

1. "The demo was the model. The six months was the harness."
2. "This is microservices architecture applied to cognitive work."
3. "Instead of deterministic functions, you are orchestrating probabilistic reasoning engines. That changes everything."
4. "Without purpose-built observability, you are flying blind in a system that is designed to be non-deterministic."
5. "The engineer's primary artifact is no longer a function that computes a result."
6. "The harness is model-agnostic by design."
7. "The teams that won were not the ones with the best models. They were the ones with the best harnesses."

---

## Writing Notes

- **Do not oversell.** Stage 2 is a small project. Its power is that it applies production patterns to a small scope — which is harder and more impressive than applying messy patterns to a large scope.
- **Be specific.** Every claim should point to a specific file, function, or architectural decision. No hand-waving.
- **Show the unglamorous parts.** The message rewriting for Mistral, the `operator.add` reducers, the InputState/InterviewState split — these are the details that prove real engineering, not demo engineering.
- **The blog is the portfolio piece.** The reader should finish thinking: "This person understands that the hard problem is the harness, and they can build one."
