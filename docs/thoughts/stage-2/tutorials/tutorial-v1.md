# Building a Two-Bot Interview System with LangGraph

A hands-on tutorial that walks through building a multi-agent conversation system where two AI personas interview each other, powered by LangGraph orchestration, Mistral AI, and Langfuse observability.

---

## What You'll Build

By the end of this tutorial, you'll have a system where two AI characters - say, a tough investigative reporter and a slippery politician - have a back-and-forth conversation on any topic you choose. The system:

- Orchestrates turn-taking between two LLM-powered bots using LangGraph
- Manages prompts externally through Langfuse (edit personas without touching code)
- Traces every LLM call for debugging and evaluation
- Streams responses to the terminal in real time
- Supports multiple persona pairings out of the box

Here's what a run looks like:

```
Two-Bot Interview: Reporter vs Politician
Preset: reporter-politician | Max turns: 3 | Session: interview-a1b2c3d4
============================================================

Enter interview topic: campaign finance reform

Topic: campaign finance reform
------------------------------------------------------------

[Reporter]: Senator, your campaign received $2.3 million from the
pharmaceutical lobby last quarter alone...

[Politician]: Well, you know, I appreciate the question, and let me
tell you about my grandmother...

[Reporter]: With respect, Senator, that doesn't address the specific
donations from...
```

## Prerequisites

- Python 3.12+
- A Mistral AI API key
- A Langfuse account (free tier works)
- Basic familiarity with Python async patterns and Pydantic

---

## Part 1: What Is LangGraph and Why Use It?

### The Problem with Linear Chains

If you've used LangChain before, you've probably built something like:

```python
chain = prompt | llm | parser
result = chain.invoke({"input": "hello"})
```

This works great for single-shot tasks: summarize this document, answer this question, extract these fields. But what happens when you need:

- **Multiple agents** taking turns in a conversation?
- **Conditional logic** - route to different nodes based on state?
- **Cycles** - agent A talks to agent B, who talks back to agent A?
- **Persistent state** that accumulates across turns?

Linear chains can't do any of this. You'd end up writing a `while` loop with a bunch of `if` statements, manually tracking state in dictionaries, and losing all the benefits of structured orchestration.

### Enter LangGraph

LangGraph is a framework for building **stateful, multi-step AI applications as graphs**. Instead of thinking about chains, you think about:

- **Nodes** - Functions that do work (call an LLM, process data, make decisions)
- **Edges** - Connections between nodes that define the flow
- **State** - A typed object that flows through the graph and accumulates results
- **Conditional edges** - Routing logic that decides where to go next

Here's the mental model:

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

This is a **cyclic graph** - something that's impossible with a simple chain. The initiator speaks, then the responder speaks, and they keep going back and forth until both have used up their turns. LangGraph manages the loop, the state, and the routing for you.

### Key LangGraph Concepts Used in This Tutorial

| Concept | What It Does | Where You'll See It |
|---------|-------------|-------------------|
| `StateGraph` | Defines a graph with typed state | Graph construction |
| `MessagesState` | Built-in state that tracks a list of messages | Base class for our state |
| `Annotated[int, operator.add]` | Reducer that accumulates values across turns | Turn counters |
| `add_conditional_edges` | Routes to different nodes based on state | Turn-limit checking |
| `InputState` vs internal state | Separates public API from implementation details | State design |
| `stream_mode="updates"` | Streams node-by-node results | CLI output |

---

## Part 2: Project Structure

The project is organized as a uv workspace package with four files, each with a single responsibility:

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

No file imports from a file that imports it back. This makes the system easy to test, reason about, and extend.

### Dependencies

```toml
# src/stage_2/pyproject.toml
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

The stack:
- **LangGraph** - Graph orchestration
- **LangChain Core** - Prompt templates, message types, runnable interface
- **LangChain Mistral** - LLM integration
- **Langfuse** - Prompt management and tracing
- **Pydantic Settings** - Type-safe environment variable loading

---

## Part 3: Configuration

Before building the graph, we need API keys. The `config.py` module loads them from a `.env` file using Pydantic Settings:

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

Your `.env` file needs four values:

```
MISTRAL_API_KEY=your-mistral-key
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

Why Pydantic Settings instead of raw `os.getenv()`? It gives you type validation at startup. If `MISTRAL_API_KEY` is missing, you get a clear error immediately - not a cryptic `None` failure three minutes into a conversation.

---

## Part 4: Designing the Personas

The `personas.py` module is pure data - no logic, no imports beyond `StrEnum`. It defines the character pairings that drive conversations:

```python
# stage_2/personas.py
from enum import StrEnum


class Preset(StrEnum):
    REPORTER_POLITICIAN = "reporter-politician"
    REPORTER_BOXER = "reporter-boxer"
    DONOR_POLITICIAN = "donor-politician"
    BARTENDER_PATRON = "bartender-patron"
```

Each preset maps to two personas, each with three fields:

```python
PERSONA_PRESETS: dict[Preset, dict[str, dict[str, str]]] = {
    Preset.REPORTER_POLITICIAN: {
        "initiator": {
            "persona_name": "Reporter",
            "persona_description": "a serious investigative journalist conducting "
                "a live television interview with high ethical standards and a "
                "reputation for tough, fair questioning",
            "persona_behavior": "You press for specifics, follow up on evasions, "
                "and cite facts. You are respectful but relentless.",
        },
        "responder": {
            "persona_name": "Politician",
            "persona_description": "a seasoned but ethically questionable "
                "politician being interviewed on live TV",
            "persona_behavior": "You deflect hard questions, pivot to talking "
                "points, use folksy anecdotes, make vague promises, and "
                "occasionally attack the media. You never directly answer "
                "uncomfortable questions.",
        },
    },
    # ... more presets
}
```

The three fields (`persona_name`, `persona_description`, `persona_behavior`) are template variables that get injected into Langfuse prompts at runtime. This separation is key: the persona **data** lives in code, but the prompt **template** that uses that data lives in Langfuse where it can be edited without redeploying.

### Why This Design Matters

Adding a new persona pairing is a **data-only change**:

1. Add a new `Preset` enum value
2. Add a new entry to `PERSONA_PRESETS`
3. Done. No graph changes, no node changes, no prompt changes.

The graph doesn't care who's talking - it just orchestrates turns. The prompts don't care which preset was chosen - they just receive variables. This is the kind of separation that makes systems maintainable.

---

## Part 5: The Graph - Where It All Comes Together

This is the core of the system. Let's walk through `graph.py` section by section.

### 5.1: Langfuse Initialization

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

Langfuse v3 uses a singleton pattern. The first `Langfuse(...)` call initializes the global client. After that, `get_client()` returns the same instance anywhere in the codebase. No need to pass the client around.

The `CallbackHandler` is a LangChain callback that automatically traces every LLM call, prompt rendering, and chain invocation. You'll attach it when running the graph.

### 5.2: State Design

This is where LangGraph's power starts to show:

```python
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

There are two important design decisions here:

**Split input vs. internal state.** `InputState` is the public API - what you pass when you invoke the graph. `InterviewState` extends it with internal bookkeeping (turn counters). The caller never needs to know about `initiator_turns` or `responder_turns`. LangGraph supports this split natively via the `input` parameter on `StateGraph`.

**Reducers with `Annotated[int, operator.add]`.** This is subtle but powerful. When a node returns `{"initiator_turns": 1}`, LangGraph doesn't *set* the value to 1 - it *adds* 1 to the current value. This is a **reducer**. Each node just says "I took one turn" and the state accumulates correctly, regardless of how many times the node has run. No manual counter management, no race conditions.

`MessagesState` itself is a built-in that gives you a `messages` list with an append reducer - new messages get appended to the list rather than replacing it.

### 5.3: The Node Factory

Instead of writing two nearly identical node functions, we use a factory that builds a node for any role:

```python
def _build_node_fn(role: str, prompt_name: str):
    turns_key = f"{role}_turns"
    name_key = f"{role}_name"
    other_role = "responder" if role == "initiator" else "initiator"
    other_name_key = f"{other_role}_name"

    def node_fn(state: InterviewState, config: RunnableConfig) -> dict:
        # ... pipeline ...
        pass

    node_fn.__name__ = role
    return node_fn


initiator = _build_node_fn("initiator", "interview/initiator")
responder = _build_node_fn("responder", "interview/responder")
```

The closure captures `role`, `prompt_name`, and derived keys like `turns_key`. The `__name__` assignment ensures LangGraph displays meaningful node names in traces and the Studio UI.

### 5.4: Inside the Node - The Five-Step Pipeline

Each node invocation follows the same five steps:

**Step 1: Fetch the prompt from Langfuse**

```python
langfuse = get_client()
lf_prompt = langfuse.get_prompt(prompt_name, type="chat")
```

This fetches the latest version of the prompt from Langfuse's server. The prompt is a chat template with variables like `{{persona_name}}`, `{{persona_description}}`, `{{persona_behavior}}`, and `{{other_persona}}`.

**Step 2: Compile with persona variables**

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

`compile()` replaces the template variables with actual values. For the Reporter in the `reporter-politician` preset, this produces a system message like: *"You are Reporter, a serious investigative journalist... You are speaking with Politician..."*

**Step 3: Rewrite message history for Mistral**

```python
from langchain_core.messages import HumanMessage as HM, AIMessage as AIM

history = []
for msg in state["messages"]:
    if isinstance(msg, HM):
        history.append(msg)
    elif isinstance(msg, AIM):
        history.append(HM(content=msg.content, name=msg.name))
```

This is the trickiest part of the system and deserves a closer look.

**The problem:** Both bots produce `AIMessage`s (they're both AI responses). But Mistral requires strict user/assistant message alternation - you can't send two `AIMessage`s in a row.

**The solution:** From each bot's perspective, the other bot's messages are *input* (things said to them), not *output* (things they said). So before calling the LLM, we convert all `AIMessage`s in the history to `HumanMessage`s. The only `AIMessage` in the conversation will be the one the current bot is about to generate.

We preserve the `name` attribute during conversion so we can still tell who said what when rendering the output.

**Step 4: Build the LLM from prompt config**

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
)
```

Two things happening here:

1. **Prompt-to-trace linking.** Setting `langchain_prompt.metadata = {"langfuse_prompt": lf_prompt}` tells the Langfuse callback handler to link this trace to the exact prompt version that produced it. In the Langfuse dashboard, you can click from a trace straight to the prompt that was used.

2. **Model config from Langfuse.** The model name and temperature come from the prompt's config object, not from code. Want to switch from `mistral-small-latest` to `mistral-medium-latest`? Change it in the Langfuse dashboard. Next run picks it up automatically.

**Step 5: Invoke and return**

```python
chain = langchain_prompt | llm
response = chain.invoke({"messages": history}, config=config)
response.name = state[name_key]

return {
    "messages": [response],
    turns_key: 1,
}
```

The node returns a partial state update. LangGraph applies reducers: the response gets *appended* to `messages` (via `MessagesState`'s built-in reducer), and `1` gets *added* to the turn counter (via our `operator.add` reducer).

### 5.5: Conditional Routing

After each bot speaks, we need to decide: does the conversation continue, or is it done?

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

Notice the cross-check pattern: after the initiator speaks, we check if the *responder* still has turns. After the responder speaks, we check if the *initiator* still has turns. This ensures each bot gets exactly `max_turns` messages.

### 5.6: Graph Assembly

All the pieces come together in a few lines:

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

Let's trace through what happens with `max_turns=3`:

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

Result: 6 messages total (3 per bot), alternating perfectly.

The last line registers a shutdown hook:

```python
atexit.register(get_client().flush)
```

This ensures any buffered Langfuse traces get sent before the process exits. Without this, you might lose the last few traces if the program exits quickly.

---

## Part 6: The CLI - Streaming It All Together

The `main.py` module provides a command-line interface that streams the conversation to the terminal:

```python
# stage_2/main.py
import argparse
import uuid
from langchain_core.messages import AIMessage, HumanMessage
from stage_2.graph import graph, get_langfuse_handler, get_langfuse_client
from stage_2.personas import Preset, PERSONA_PRESETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-Bot Interview")
    parser.add_argument(
        "--preset",
        choices=[p.value for p in Preset],
        default=Preset.REPORTER_POLITICIAN.value,
        help="Persona pairing (default: reporter-politician)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=3,
        help="Max messages per bot (default: 3)",
    )
    return parser.parse_args()
```

The CLI accepts two arguments: which persona preset to use, and how many turns each bot gets.

### Running the Graph with Streaming

The core of the CLI is the streaming loop:

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
            "max_turns": max_turns,
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

Key details:

- **`stream_mode="updates"`** gives us node-by-node output. Each `update` is a dict like `{"initiator": {"messages": [AIMessage(...)]}}`. This lets us print each response as soon as it's ready, rather than waiting for the entire conversation to finish.

- **`langfuse_session_id`** in the config metadata groups all traces from this run under one session in the Langfuse dashboard. This makes it easy to find and review a complete conversation.

- **`flush()`** at the end ensures all traces are sent to Langfuse before the process exits.

### Running It

```bash
# Default: Reporter vs Politician, 3 turns each
make chat SCOPE=2

# Custom preset and turns
make chat SCOPE=2 ARGS="--preset bartender-patron --max-turns 5"

# Or directly
uv run --package stage-2 python -m stage_2.main --preset donor-politician --max-turns 2
```

---

## Part 7: Langfuse Prompt Management

One of the most powerful aspects of this system is that **prompts live outside the code**. Here's how to set it up.

### Creating Prompts in Langfuse

You need two chat prompts in your Langfuse project:

**Prompt 1: `interview/initiator`**

Type: Chat
Messages:
```
[system] You are {{persona_name}}, {{persona_description}}.
You are having a conversation with {{other_persona}}.
{{persona_behavior}}
Keep your responses to 2-3 paragraphs. Stay in character at all times.
```

Config:
```json
{
  "model": "mistral-small-latest",
  "temperature": 0.9
}
```

**Prompt 2: `interview/responder`**

Same structure, potentially with different instructions (e.g., "Respond to what was just said to you" instead of "Drive the conversation forward").

### The Variables

Each prompt receives four variables at runtime:

| Variable | Source | Example |
|----------|--------|---------|
| `{{persona_name}}` | `PERSONA_PRESETS[preset][role]["persona_name"]` | "Reporter" |
| `{{persona_description}}` | `PERSONA_PRESETS[preset][role]["persona_description"]` | "a serious investigative journalist..." |
| `{{persona_behavior}}` | `PERSONA_PRESETS[preset][role]["persona_behavior"]` | "You press for specifics..." |
| `{{other_persona}}` | `state["responder_name"]` or `state["initiator_name"]` | "Politician" |

### Why This Matters

With prompts in Langfuse, you can:

1. **Edit prompts without redeploying.** Change the system instructions, adjust the tone, add constraints - the next run picks up the new version automatically.

2. **Version everything.** Every edit creates a new version. You can diff versions, see what changed, and roll back if something breaks.

3. **Change the model from a dashboard.** The `config` object on each prompt specifies the model and temperature. Switch from `mistral-small-latest` to `mistral-medium-latest` without touching code.

4. **Link traces to prompt versions.** When you review a conversation in Langfuse, each trace shows exactly which prompt version produced it. If a conversation went off the rails, you can see the exact prompt that was active.

---

## Part 8: LangGraph Studio

The project includes a `langgraph.json` file that registers the graph for LangGraph Studio:

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

Run it with:

```bash
make dev SCOPE=2
```

LangGraph Studio gives you a visual representation of the graph, lets you step through executions, inspect state at each node, and replay conversations. It's invaluable for debugging conditional routing and understanding how state evolves across turns.

---

## Part 9: Architecture Recap

Let's zoom out and look at the full picture.

### Separation of Concerns

| Component | File | Changes When... |
|-----------|------|-----------------|
| API keys | `config.py` | Infrastructure changes |
| Character data | `personas.py` | New persona added |
| Orchestration | `graph.py` | Graph structure changes |
| UX | `main.py` | CLI behavior changes |
| Prompt text | Langfuse dashboard | Persona behavior tweaked |
| Model selection | Langfuse prompt config | Model/temperature changed |

### What's In Code vs. What's External

| In Code | External |
|---------|----------|
| Graph structure (nodes, edges, routing) | Prompt templates |
| State schema and reducers | Model selection |
| Persona data (names, descriptions) | Temperature settings |
| Message rewriting logic | Prompt version history |
| Streaming and CLI | Trace storage and analysis |

### Patterns Worth Noting

1. **Node Factory.** One function builds both nodes. Zero duplication, easy to add a third role if needed.

2. **Reducers.** `Annotated[int, operator.add]` means each node just returns `1` and the framework handles accumulation. No shared mutable state.

3. **Input/Internal State Split.** Callers see `InputState`. The graph internally uses `InterviewState`. Clean API boundary.

4. **Message Rewriting.** A pragmatic solution to Mistral's alternation requirement that doesn't leak into the graph's public interface.

5. **External Config.** Model, temperature, and prompt text are all outside the codebase. The code defines *structure*; external systems define *behavior*.

---

## Part 10: Extending the System

### Add a New Persona Preset

This is a data-only change - no graph or prompt modifications needed:

```python
# In personas.py
class Preset(StrEnum):
    # ... existing presets ...
    TEACHER_STUDENT = "teacher-student"

PERSONA_PRESETS[Preset.TEACHER_STUDENT] = {
    "initiator": {
        "persona_name": "Professor",
        "persona_description": "a Socratic philosophy professor...",
        "persona_behavior": "You ask probing questions...",
    },
    "responder": {
        "persona_name": "Student",
        "persona_description": "a freshman philosophy student...",
        "persona_behavior": "You try to answer but often get confused...",
    },
}
```

Then run it:

```bash
make chat SCOPE=2 ARGS="--preset teacher-student"
```

### Change Persona Behavior

Edit the prompt in the Langfuse dashboard. No code changes, no redeployment. The next run fetches the updated prompt automatically.

### Switch Models

Update the prompt config in Langfuse:

```json
{
  "model": "mistral-medium-latest",
  "temperature": 0.7
}
```

### Add More Turns

```bash
make chat SCOPE=2 ARGS="--max-turns 10"
```

---

## Summary

This tutorial walked through building a multi-agent conversation system with LangGraph. The key takeaway is that **the hard engineering is the harness, not the model**. The LLM calls are the simplest part. The real work is in:

- **Orchestration** - Managing turns, routing, and state accumulation
- **Message handling** - Rewriting message types to satisfy model requirements
- **Prompt management** - Keeping prompts external and version-controlled
- **Observability** - Tracing every call for debugging and evaluation
- **Clean architecture** - Separating concerns so changes are isolated

The entire system is four Python files totaling roughly 300 lines of code. But those 300 lines demonstrate production patterns: factory functions, typed state with reducers, conditional routing, external configuration, full tracing, and version-controlled prompts. These are the same patterns you'd use to build a much larger system - the scale changes, but the architecture doesn't.
