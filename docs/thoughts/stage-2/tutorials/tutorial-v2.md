# Building the Harness, Not the Demo: A Two-Bot Interview System with LangGraph

Everyone builds the impressive AI demo in a weekend. Then they spend six months making it work in production. The demo is the model. The six months is the harness — the orchestration, the observability, the prompt management, the state design, the edge cases that only surface when you wire real systems together.

What if you practiced harness-first thinking even on a small project?

That's what this tutorial is about. We'll build a system where two AI personas — an investigative reporter and a slippery politician, or a weary bartender and a drunk patron at 1 AM — take turns speaking on any topic you choose. The conversations are entertaining. But the interesting part isn't what the bots say. It's the system that makes them say it: a LangGraph state graph with typed state and conditional routing, Langfuse prompt management with version-controlled prompts fetched at runtime, and full observability tracing every LLM call with session grouping and prompt-version linking.

Here's what a run looks like:

```
Two-Bot Interview: Reporter vs Politician
Preset: reporter-politician | Max turns: 3 | Session: interview-a1b2c3d4
============================================================

Enter interview topic: campaign finance reform

Topic: campaign finance reform
------------------------------------------------------------

[Reporter]: Senator, your campaign received $2.3 million from the
pharmaceutical lobby last quarter alone. Can you explain what policy
commitments, if any, accompanied those contributions?

[Politician]: Well, you know, I appreciate the question, and let me
tell you — my grandmother grew up in a small town where people looked
out for each other, and that's the kind of values I bring to...

[Reporter]: With respect, Senator, that doesn't address the specific
donations. Your FEC filings show...
```

The system is four Python files totaling roughly 300 lines. But those 300 lines demonstrate production patterns: factory-built nodes, typed state with reducers, conditional routing, externalized prompt management, full tracing, and version-controlled configuration. These are the same patterns you'd use for a multi-agent system in healthcare or legal — the domain changes, the harness doesn't.

**Prerequisites:** Python 3.12+, a Mistral AI API key, a Langfuse account (free tier works), and basic familiarity with Pydantic.

---

## Part 1: What Is LangGraph and Why Use It Here?

If you've used LangChain, you've built chains:

```python
chain = prompt | llm | parser
result = chain.invoke({"input": "hello"})
```

This works for single-shot tasks. Summarize a document. Answer a question. Extract some fields. But what happens when you need two agents taking turns? Conditional logic that routes to different nodes based on state? Cycles — agent A talks to agent B, who talks back to agent A? Persistent state that accumulates across turns?

Chains can't do this. You'd end up writing a `while` loop with a pile of `if` statements, manually tracking state in dictionaries, and losing all the benefits of structured orchestration.

LangGraph is a framework for building **stateful, multi-step AI applications as graphs**. Instead of thinking in chains, you think in:

- **Nodes** — functions that do work (call an LLM, process data, make decisions)
- **Edges** — connections between nodes that define flow
- **State** — a typed object that flows through the graph and accumulates results
- **Conditional edges** — routing logic that decides where to go next

Here's the mental model for what we're building:

```
         ┌────────────┐
         │   START     │
         └─────┬──────┘
               │
               ▼
         ┌────────────┐
    ┌───►│ Initiator  │
    │    └─────┬──────┘
    │          │
    │          ▼ (conditional)
    │    ┌────────────┐
    │    │ Responder  │
    │    └─────┬──────┘
    │          │
    │          ▼ (conditional)
    │          │
    └──────────┘  ← loop back if turns remain
               │
               ▼
         ┌────────────┐
         │    END      │
         └────────────┘
```

This is a **cyclic graph** — impossible with a linear chain. The initiator speaks, then the responder speaks, and they loop until both have exhausted their turns. LangGraph manages the loop, the state, and the routing. You define the structure; the framework executes it.

---

## Part 2: Project Structure — Four Files, Four Jobs

> "This is microservices architecture applied to cognitive work. The patterns are familiar to anyone who has built distributed systems: service decomposition, message passing, state management."

The system decomposes into four files, each with a single responsibility:

```
src/stage_2/
├── pyproject.toml          # Dependencies
└── stage_2/
    ├── __init__.py
    ├── config.py            # Environment variable loading
    ├── personas.py          # Persona data (pure data, no logic)
    ├── graph.py             # LangGraph orchestration (the brain)
    └── main.py              # CLI entry point (streaming output)
```

The dependency flow is strictly one-directional:

```
.env → config.py → graph.py ← personas.py
                       ↓
                    main.py
                       ↓
                     User
```

No circular imports. Each file can be understood in isolation. Change persona definitions — edit `personas.py` only. Change prompt wording — edit the Langfuse dashboard only (no files at all). Change the graph structure — edit `graph.py` only. Change CLI behavior — edit `main.py` only.

The stack:

```toml
[project]
name = "stage-2"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-mistralai>=1.1.1",
    "langfuse>=3.12.1",
    "langgraph>=1.0.7",
    "pydantic-settings>=2.12.0",
]
```

- **LangGraph** — state graph orchestration
- **LangChain Core** — prompt templates, message types, runnable interface
- **LangChain Mistral** — LLM integration
- **Langfuse** — dual role: prompt management *and* observability tracing
- **Pydantic Settings** — type-safe environment variable loading

---

## Part 3: Configuration — Fail Fast, Fail Clearly

Before building the graph, we need API keys. `config.py` loads them from a `.env` file using Pydantic Settings:

```python
# stage_2/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mistral AI
    mistral_api_key: str

    # Langfuse
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_base_url: str = "https://us.cloud.langfuse.com"


def get_settings() -> Settings:
    return Settings()
```

Your `.env` needs four values:

```
MISTRAL_API_KEY=your-mistral-key
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

Why Pydantic Settings instead of `os.getenv()`? Type validation at startup. If `MISTRAL_API_KEY` is missing, you get a clear error before anything runs — not a cryptic `None` failure three minutes into a conversation when the first LLM call fires. Fail fast, fail clearly.

---

## Part 4: The Personas — Pure Data, Maximum Flexibility

`personas.py` is the simplest file in the system and one of the most important design decisions. It contains zero logic — just a `StrEnum` and a dictionary:

```python
# stage_2/personas.py
from enum import StrEnum


class Preset(StrEnum):
    REPORTER_POLITICIAN = "reporter-politician"
    REPORTER_BOXER = "reporter-boxer"
    DONOR_POLITICIAN = "donor-politician"
    BARTENDER_PATRON = "bartender-patron"
```

Each preset maps to two personas. Each persona has three fields that become template variables at runtime:

```python
PERSONA_PRESETS: dict[Preset, dict[str, dict[str, str]]] = {
    Preset.REPORTER_POLITICIAN: {
        "initiator": {
            "persona_name": "Reporter",
            "persona_description": "a serious investigative journalist "
                "conducting a live television interview with high ethical "
                "standards and a reputation for tough, fair questioning",
            "persona_behavior": "You press for specifics, follow up on "
                "evasions, and cite facts. You are respectful but relentless.",
        },
        "responder": {
            "persona_name": "Politician",
            "persona_description": "a seasoned but ethically questionable "
                "politician being interviewed on live TV",
            "persona_behavior": "You deflect hard questions, pivot to "
                "talking points, use folksy anecdotes, make vague promises, "
                "and occasionally attack the media. You never directly "
                "answer uncomfortable questions.",
        },
    },
    Preset.BARTENDER_PATRON: {
        "initiator": {
            "persona_name": "Bartender",
            "persona_description": "a weary, seen-it-all bartender working "
                "the late shift at a dive bar",
            "persona_behavior": "You listen, offer unsolicited life advice, "
                "make dry observations, and occasionally cut off the patron "
                "or change the subject. You've heard every sad story before.",
        },
        "responder": {
            "persona_name": "Patron",
            "persona_description": "a drunk patron at a dive bar at 1 AM "
                "who clearly has something on their mind",
            "persona_behavior": "You ramble, go on tangents, get emotional, "
                "contradict yourself, and occasionally order another drink "
                "mid-sentence. You're convinced this is the most important "
                "conversation of your life.",
        },
    },
    # ... two more presets
}
```

The `persona_behavior` field is where characters come alive. These aren't polite instructions — they're specific behavioral directives that create natural tension between the two roles. The Politician *never* directly answers uncomfortable questions. The Patron orders drinks *mid-sentence*. The Boxer *threatens to flip the table*. This tension is what makes conversations interesting without any special orchestration logic.

The three fields — `persona_name`, `persona_description`, `persona_behavior` — map directly to template variables in the Langfuse prompts. The persona data lives in code. The prompt template that consumes it lives in Langfuse. Two independent levers for tuning behavior.

Adding a new preset is a data-only change. Add the enum value, add the dictionary entry — done. No graph changes, no node changes, no prompt changes. The graph doesn't care who's talking. The prompts don't care which preset was chosen. Each layer is ignorant of the others' details, which is exactly the kind of separation that makes systems maintainable.

---

## Part 5: The Graph — Orchestration That Compounds

This is the core of the system. `graph.py` is 184 lines that wire together state management, prompt fetching, message handling, LLM invocation, and conditional routing. Let's walk through it.

### Langfuse Initialization

```python
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

_settings = get_settings()
Langfuse(
    public_key=_settings.langfuse_public_key,
    secret_key=_settings.langfuse_secret_key,
    host=_settings.langfuse_base_url,
)

def get_langfuse_client() -> Langfuse:
    return get_client()

def get_langfuse_handler() -> CallbackHandler:
    return CallbackHandler()
```

Langfuse v3 uses a singleton pattern. The first `Langfuse(...)` call at module load initializes the global client. After that, `get_client()` returns the same instance anywhere in the codebase — no need to pass the client around, no risk of creating multiple connections.

The `CallbackHandler` is a LangChain callback that automatically traces every LLM call, prompt rendering, and chain invocation. You'll attach it once when running the graph, and every node execution gets traced for free.

### State Design — The API Boundary

> "Instead of deterministic functions, you are orchestrating probabilistic reasoning engines. That changes everything."

State design is where LangGraph's power starts to show:

```python
import operator
from typing import Annotated
from langgraph.graph import MessagesState

class InputState(MessagesState):
    """What the caller provides."""
    max_turns: int
    preset: Preset
    initiator_name: str
    responder_name: str


class InterviewState(InputState):
    """What the graph tracks internally."""
    initiator_turns: Annotated[int, operator.add]
    responder_turns: Annotated[int, operator.add]
```

Two design decisions here, both deliberate.

**Split input vs. internal state.** `InputState` is the public API — what you pass when you invoke the graph. `InterviewState` extends it with internal bookkeeping that callers should never touch. The graph is compiled with `StateGraph(InterviewState, input=InputState)`, so LangGraph enforces the split. A caller cannot accidentally set `initiator_turns` to 5. The graph controls its own internals.

This is the LangGraph equivalent of separating your API contract from your internal implementation state.

**Reducers with `Annotated[int, operator.add]`.** When a node returns `{"initiator_turns": 1}`, LangGraph doesn't *set* the value to 1 — it *adds* 1 to the current value. This is a **reducer**. Each node just says "I took one turn" and the state accumulates correctly, regardless of how many times the node has run. No reading the current count, incrementing, and writing back. No off-by-one bugs. No race conditions.

`MessagesState` itself provides a `messages` list with an append reducer — new messages get appended rather than replacing the list.

### The Node Factory — Build Once, Use Twice

Instead of writing two nearly identical functions, a factory builds both nodes from the same blueprint:

```python
def _build_node_fn(role: str, prompt_name: str):
    turns_key = f"{role}_turns"
    name_key = f"{role}_name"
    other_role = "responder" if role == "initiator" else "initiator"
    other_name_key = f"{other_role}_name"

    def node_fn(state: InterviewState, config: RunnableConfig) -> dict:
        # ... 5-step pipeline ...
        return {"messages": [response], turns_key: 1}

    node_fn.__name__ = role
    return node_fn

initiator = _build_node_fn("initiator", "interview/initiator")
responder = _build_node_fn("responder", "interview/responder")
```

The closure captures `role`, `prompt_name`, and derived keys. The `__name__` assignment ensures LangGraph displays meaningful names in traces and the Studio UI. Both nodes run the same 5-step pipeline — they differ only in which persona they load, which prompt they fetch, and which turn counter they increment.

In a larger system with five or ten agent roles, this factory prevents the codebase from growing linearly with agent count. Add a role by calling the factory with new parameters, not by copying and modifying a function.

### Inside the Node — The Five-Step Pipeline

Each node invocation follows the same sequence:

**Step 1: Fetch the prompt from Langfuse.**

```python
langfuse = get_client()
lf_prompt = langfuse.get_prompt(prompt_name, type="chat")
```

This fetches the latest version of the prompt from Langfuse. The prompt is a chat template with variables — `{{persona_name}}`, `{{persona_description}}`, `{{persona_behavior}}`, `{{other_persona}}`.

**Step 2: Compile with persona variables.**

```python
persona = PERSONA_PRESETS[state["preset"]][role]
compiled_messages = lf_prompt.compile(
    persona_name=persona["persona_name"],
    persona_description=persona["persona_description"],
    persona_behavior=persona["persona_behavior"],
    other_persona=state[other_name_key],
)
system_content = compiled_messages[0]["content"]
```

`compile()` replaces template variables with actual values. For the Reporter in `reporter-politician`, this produces a system message like: *"You are Reporter, a serious investigative journalist... You are speaking with Politician... You press for specifics, follow up on evasions, and cite facts."*

**Step 3: Rewrite message history for Mistral.**

This is the trickiest part of the system — the kind of problem that never surfaces in a single-bot demo.

```python
from langchain_core.messages import HumanMessage as HM, AIMessage as AIM

history = []
for msg in state["messages"]:
    if isinstance(msg, HM):
        history.append(msg)
    elif isinstance(msg, AIM):
        history.append(HM(content=msg.content, name=msg.name))
```

**The problem:** Both bots produce `AIMessage`s — they're both AI responses. But Mistral requires strict user/assistant message alternation. You can't send two `AIMessage`s in a row. In a two-bot system, the shared message history naturally has consecutive `AIMessage`s (initiator speaks, then responder speaks, both as `AIMessage`).

**The solution:** From each bot's perspective, the other bot's messages are *input* (things said to them), not *output* (things they said). So before calling the LLM, each node converts all `AIMessage`s in history to `HumanMessage`s. The only `AIMessage` in the conversation will be the one the current bot is about to generate.

We preserve the `name` attribute during conversion so we can still identify speakers in the output.

This is the kind of unglamorous systems engineering that makes multi-agent systems actually work. The model API has opinions about conversation structure, and the harness must accommodate them.

**Step 4: Build the LLM from prompt config.**

```python
langchain_prompt = ChatPromptTemplate.from_messages([
    ("system", system_content),
    MessagesPlaceholder("messages"),
])
langchain_prompt.metadata = {"langfuse_prompt": lf_prompt}

model_config = lf_prompt.config or {}
llm = ChatMistralAI(
    model=model_config.get("model", "mistral-small-latest"),
    temperature=model_config.get("temperature", 0.9),
    api_key=_settings.mistral_api_key,
)
```

Two things happening here.

Setting `langchain_prompt.metadata = {"langfuse_prompt": lf_prompt}` tells the Langfuse callback handler to link this trace to the exact prompt version that produced it. In the dashboard, you can click from any trace straight to the prompt text that was active. This is prompt-to-trace linking — more on this in the observability section.

The model name and temperature come from the Langfuse prompt's `config` object, not from code. Want to switch from `mistral-small-latest` to `mistral-medium-latest`? Change it in the dashboard. Next run picks it up automatically. The code has sensible defaults but defers to external configuration.

**Step 5: Invoke and return.**

```python
chain = langchain_prompt | llm
response = chain.invoke({"messages": history}, config=config)
response.name = state[name_key]

return {
    "messages": [response],
    turns_key: 1,
}
```

The node returns a partial state update. LangGraph applies reducers: the response gets *appended* to `messages` (via `MessagesState`'s built-in reducer), and `1` gets *added* to the turn counter (via our `operator.add` reducer). Setting `response.name` ensures the `AIMessage` carries the persona's display name — the CLI can read it directly without knowing which node produced the message.

### Conditional Routing — Deterministic Flow Control

After each bot speaks, a routing function decides whether the conversation continues:

```python
def after_initiator(state: InterviewState) -> str:
    if state["responder_turns"] < state["max_turns"]:
        return "responder"
    return END


def after_responder(state: InterviewState) -> str:
    if state["initiator_turns"] < state["max_turns"]:
        return "initiator"
    return END
```

Notice the cross-check: after the initiator speaks, we check if the *responder* still has turns. After the responder speaks, we check if the *initiator* still has turns. This ping-pong pattern guarantees each bot gets exactly `max_turns` messages and the graph always terminates.

Deterministic termination is a reliability property. There is no infinite loop. There is no ambiguous stopping condition. The graph algebra enforces it.

### Graph Assembly — Seven Lines

All the pieces come together:

```python
def create_graph(input_schema: type = InputState) -> CompiledStateGraph:
    builder = StateGraph(InterviewState, input=input_schema)
    builder.add_node("initiator", initiator)
    builder.add_node("responder", responder)
    builder.add_edge(START, "initiator")
    builder.add_conditional_edges("initiator", after_initiator, ["responder", END])
    builder.add_conditional_edges("responder", after_responder, ["initiator", END])
    return builder.compile()

graph = create_graph()
```

Seven lines. Two nodes, three edges. The simplicity is the point — the orchestration layer is minimal and readable, with complexity pushed into the node factory and state design where it belongs.

Let's trace through `max_turns=3` to see the full execution:

1. `START` -> `initiator` (initiator_turns: 0 -> 1)
2. `after_initiator`: responder_turns (0) < 3? Yes -> `responder`
3. `responder` speaks (responder_turns: 0 -> 1)
4. `after_responder`: initiator_turns (1) < 3? Yes -> `initiator`
5. `initiator` speaks (initiator_turns: 1 -> 2)
6. `after_initiator`: responder_turns (1) < 3? Yes -> `responder`
7. `responder` speaks (responder_turns: 1 -> 2)
8. `after_responder`: initiator_turns (2) < 3? Yes -> `initiator`
9. `initiator` speaks (initiator_turns: 2 -> 3)
10. `after_initiator`: responder_turns (2) < 3? Yes -> `responder`
11. `responder` speaks (responder_turns: 2 -> 3)
12. `after_responder`: initiator_turns (3) < 3? No -> `END`

Six messages total — three per bot, alternating perfectly.

One last detail:

```python
atexit.register(get_client().flush)
```

This registers a shutdown hook to flush any buffered Langfuse traces before the process exits. Demo code doesn't worry about trace flushing. Production code does.

---

## Part 6: Observability — Seeing Inside the Conversation

> "Without purpose-built observability, you are flying blind in a system that is designed to be non-deterministic."

The interview system is non-deterministic by design — two bots riffing on an open-ended topic. But every run is fully traced. Three mechanisms make this work.

### Automatic Instrumentation

A single `CallbackHandler` does all the work:

```python
langfuse_handler = get_langfuse_handler()

for update in graph.stream(
    {... input state ...},
    config={
        "callbacks": [langfuse_handler],
        "metadata": {"langfuse_session_id": session_id},
    },
    stream_mode="updates",
):
```

LangGraph's execution engine automatically calls the handler's hooks on every node invocation. No manual span creation. No decorators. No `with trace():` blocks.

For each of the six LLM calls in a 3-turn interview, Langfuse captures: the full compiled system prompt, the rewritten message history, the response, token counts (input and output), latency, and model version. Two lines of setup code. Total visibility.

### Session Grouping

`main.py` generates a unique session ID per run:

```python
session_id = f"interview-{uuid.uuid4().hex[:8]}"
```

This ID, passed through `config["metadata"]["langfuse_session_id"]`, groups all six traces from one interview under a single session in the dashboard. Without it, you'd have six disconnected traces. With it, you see the full Reporter-vs-Politician transcript as one unit — total cost, total latency, full conversation timeline.

### Prompt-to-Trace Linking

This is the mechanism that closes the loop between prompt management and observability.

Remember the metadata line from the node pipeline?

```python
langchain_prompt.metadata = {"langfuse_prompt": lf_prompt}
```

This creates a bidirectional link:
- **From trace to prompt**: Open a trace in Langfuse, see which prompt version produced it, read the exact text
- **From prompt to traces**: Open a prompt version, see all traces that used it, evaluate quality

Here's a concrete debugging story. You run ten interviews over two days. Tuesday's conversations feel off — the Politician is giving straight answers instead of deflecting. You open Langfuse, filter sessions by preset, compare Tuesday's traces to Monday's. Tuesday's traces all link to prompt version v4. Monday's linked to v3. You diff v4 against v3 in the Langfuse UI and find someone removed the instruction "You never directly answer uncomfortable questions." You roll back to v3. Next interview: the Politician deflects again. Total debugging time: three minutes. No code was touched.

This is the data flywheel beginning to turn. Traces aren't just logged — they're connected to the configuration that produced them, making systematic improvement possible.

---

## Part 7: Prompt Management as Infrastructure

> "The engineer's primary artifact is no longer a function that computes a result. It is a harness that orchestrates, validates, and monitors something else computing the result."

Most projects hardcode prompts as Python strings. Stage 2 treats prompts as externalized, versioned, runtime-fetched infrastructure.

### Prompts Outside Code

The two system prompts — `interview/initiator` and `interview/responder` — live in Langfuse as managed chat prompts. They're templates with four variables:

```
[system] You are {{persona_name}}, {{persona_description}}.
You are having a conversation with {{other_persona}}.
{{persona_behavior}}
Keep your responses to 2-3 paragraphs. Stay in character at all times.
```

On every node invocation, the graph fetches the latest version, compiles it with persona variables, and uses the result. Editing a prompt is a dashboard operation — open Langfuse, change the wording, save. The next graph invocation picks up the new version automatically. No code change. No PR. No redeployment.

### Model Config Outside Code

Each Langfuse prompt carries a `config` object:

```json
{
  "model": "mistral-small-latest",
  "temperature": 0.9
}
```

The node factory reads this config when constructing the LLM:

```python
model_config = lf_prompt.config or {}
llm = ChatMistralAI(
    model=model_config.get("model", "mistral-small-latest"),
    temperature=model_config.get("temperature", 0.9),
    api_key=_settings.mistral_api_key,
)
```

Switching from `mistral-small-latest` to `mistral-medium-latest`, or tuning temperature from 0.9 to 0.7, is a dashboard change. The model is a swappable component controlled from outside the codebase.

### Version History

Every prompt edit in Langfuse creates a new version. Combined with prompt-to-trace linking, you get a full audit trail: this conversation used prompt v3, which had this exact text, and produced these results. You can diff versions, evaluate which performed better, and roll back in seconds.

The code defines *structure*. External systems define *behavior*. That boundary is what makes the system maintainable.

---

## Part 8: The CLI — Streaming the Conversation

The final layer is `main.py`, which provides a CLI that streams the conversation to the terminal in real time:

```python
def run_interview() -> None:
    args = parse_args()
    preset_key = Preset(args.preset)
    preset = PERSONA_PRESETS[preset_key]

    initiator_name = preset["initiator"]["persona_name"]
    responder_name = preset["responder"]["persona_name"]
    session_id = f"interview-{uuid.uuid4().hex[:8]}"

    topic = input("\nEnter interview topic: ").strip()

    langfuse_handler = get_langfuse_handler()

    for update in graph.stream(
        {
            "messages": [HumanMessage(content=f"Interview topic: {topic}")],
            "max_turns": args.max_turns,
            "preset": preset_key,
            "initiator_name": initiator_name,
            "responder_name": responder_name,
        },
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
        stream_mode="updates",
    ):
        for node_name, node_output in update.items():
            if "messages" not in node_output:
                continue
            for msg in node_output["messages"]:
                if isinstance(msg, AIMessage):
                    speaker = msg.name or node_name
                    print(f"\n[{speaker}]: {msg.content}")

    get_langfuse_client().flush()
```

`stream_mode="updates"` is the key. Instead of waiting for the entire conversation to complete and returning the final state, LangGraph yields output node-by-node. Each `update` is a dictionary like `{"initiator": {"messages": [AIMessage(...)]}}`. The CLI prints each response the moment it arrives — the user watches a live conversation unfold, not a data dump.

The `msg.name` detail is worth calling out. Each node sets `response.name = state[name_key]` before returning, so every `AIMessage` carries the persona's display name. The streaming loop doesn't need to know which node produced the message — it reads the name from the message itself. Self-describing data makes the rendering code simple and correct.

Run it:

```bash
# Default: Reporter vs Politician, 3 turns each
make chat SCOPE=2

# Custom preset and turns
make chat SCOPE=2 ARGS="--preset bartender-patron --max-turns 5"

# Or directly with uv
uv run --package stage-2 python -m stage_2.main --preset donor-politician --max-turns 2
```

---

## Part 9: LangGraph Studio

The project includes a `langgraph.json` that registers the graph for LangGraph Studio:

```json
{
  "dependencies": ["./src/stage_1", "./src/stage_2"],
  "graphs": {
    "chatbot_simple": "./src/stage_1/stage_1/graph.py:graph",
    "talking_chatbots": "./src/stage_2/stage_2/graph.py:graph"
  },
  "env": ".env"
}
```

Run it with `make dev SCOPE=2`. Studio gives you a visual representation of the graph, lets you step through executions, inspect state at each node, and replay conversations. It's invaluable for debugging conditional routing and understanding how state evolves across turns — especially when you're trying to verify that the cross-check pattern in the conditional edges produces exactly the right number of messages.

---

## Part 10: The Full Picture

Let's zoom out.

### What's In Code vs. What's External

| In Code | External |
|---------|----------|
| Graph structure (nodes, edges, routing) | Prompt templates (Langfuse) |
| State schema and reducers | Model selection (Langfuse prompt config) |
| Persona data (names, descriptions, behavior) | Temperature settings (Langfuse prompt config) |
| Message rewriting logic | Prompt version history (Langfuse) |
| Streaming and CLI | Trace storage and analysis (Langfuse) |

### When Each File Changes

| What Changed | File to Edit |
|-------------|-------------|
| New persona pairing | `personas.py` only |
| Prompt wording or model | Langfuse dashboard only |
| API keys or Langfuse URL | `.env` only |
| CLI behavior | `main.py` only |
| Graph structure | `graph.py` only |

### Patterns Worth Internalizing

**Node factory.** One function builds both nodes. Zero duplication. In a larger system, adding a new role is a function call, not a copy-paste.

**Reducers.** `Annotated[int, operator.add]` means each node returns `1` and the framework handles accumulation. No shared mutable state, no manual counting, no coordination between nodes.

**Input/internal state split.** Callers see `InputState`. The graph internally uses `InterviewState`. Clean API boundary enforced by the framework.

**Message rewriting.** A pragmatic solution to Mistral's alternation requirement that lives inside the node pipeline and doesn't leak into the graph's public interface. The kind of problem that only surfaces when you wire a second agent into the loop.

**External config.** Model, temperature, and prompt text are all outside the codebase. Code defines structure; external systems define behavior. Two independent levers for iteration.

### Extending the System

**New preset.** Add the enum value, add the dictionary entry in `personas.py`. No other files change.

**Different model.** Update the `config` on the Langfuse prompt. No code changes.

**More turns.** `make chat SCOPE=2 ARGS="--max-turns 10"`.

**New persona behavior.** Edit the Langfuse prompt template. Next run picks it up.

---

## Closing: The Harness Is the Skill

This is a small project. Two bots having a conversation. But the engineering behind it is the engineering that scales:

- A state graph with typed state, factory-built nodes, and conditional edges for deterministic flow control
- Observability from the first commit — every LLM call traced, every session grouped, every prompt version linked to the traces it produced
- Prompt management as infrastructure — prompts, model config, and persona behavior all externalized and versioned, editable from a dashboard without touching code
- Clean separation of concerns — four files, four responsibilities, one-directional dependencies

These patterns don't change when the domain changes. Replace "Reporter" with "Triage Agent" and "Politician" with "Diagnostic Agent." Replace persona presets with clinical workflow configs. Replace interview topics with patient intake data. The orchestration is identical. The observability is identical. The prompt management is identical.

The harness transfers. The harness is the skill.
