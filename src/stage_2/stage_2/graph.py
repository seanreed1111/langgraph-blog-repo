"""Two-bot interview graph with Langfuse prompt management."""

import atexit
import operator
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_mistralai import ChatMistralAI
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph

from stage_2.config import get_settings
from stage_2.personas import Preset, PERSONA_PRESETS

# ---------------------------------------------------------------------------
# Langfuse singleton (v3 pattern — same as stage_1)
# ---------------------------------------------------------------------------
_settings = get_settings()
Langfuse(
    public_key=_settings.langfuse_public_key,
    secret_key=_settings.langfuse_secret_key,
    host=_settings.langfuse_base_url,
)


def get_langfuse_client() -> Langfuse:
    """Get Langfuse singleton client instance."""
    return get_client()


def get_langfuse_handler() -> CallbackHandler:
    """Get Langfuse callback handler for tracing."""
    return CallbackHandler()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class InputState(MessagesState):
    """Input schema for the two-bot interview graph."""

    max_turns: int  # max messages per bot (default 3)
    preset: Preset  # persona preset enum (e.g., Preset.REPORTER_POLITICIAN)
    initiator_name: str  # display name (e.g., "Reporter")
    responder_name: str  # display name (e.g., "Politician")


class InterviewState(InputState):
    """Full state for the two-bot interview graph."""

    initiator_turns: Annotated[int, operator.add]  # how many times initiator has spoken
    responder_turns: Annotated[int, operator.add]  # how many times responder has spoken


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------
def _build_node_fn(role: str, prompt_name: str):
    """Return a node function for the given role ('initiator' or 'responder').

    Each call to the returned function:
    1. Fetches the Langfuse chat prompt by name
    2. Compiles it with persona variables from the preset
    3. Builds a ChatPromptTemplate with langfuse_prompt metadata (for trace linking)
    4. Chains prompt | LLM and invokes with conversation history
    5. Returns partial state update with the response and incremented turn count
    """
    turns_key = f"{role}_turns"
    name_key = f"{role}_name"
    other_role = "responder" if role == "initiator" else "initiator"
    other_name_key = f"{other_role}_name"

    def node_fn(state: InterviewState, config: RunnableConfig) -> dict:
        # 1. Fetch prompt from Langfuse
        langfuse = get_client()
        lf_prompt = langfuse.get_prompt(prompt_name, type="chat")

        # 2. Compile with persona variables
        persona = PERSONA_PRESETS[state["preset"]][role]
        compiled_messages = lf_prompt.compile(
            persona_name=persona["persona_name"],
            persona_description=persona["persona_description"],
            persona_behavior=persona["persona_behavior"],
            other_persona=state[other_name_key],
        )
        # compiled_messages is a list of dicts: [{"role": "system", "content": "..."}]
        system_content = compiled_messages[0]["content"]

        # 3. Build LangChain prompt with MessagesPlaceholder for history
        # Note: Mistral requires strict user/assistant alternation, so we convert
        # the other bot's AI messages to HumanMessages from this bot's perspective
        from langchain_core.messages import HumanMessage as HM, AIMessage as AIM

        history = []
        for msg in state["messages"]:
            if isinstance(msg, HM):
                # Keep HumanMessages as-is
                history.append(msg)
            elif isinstance(msg, AIM):
                # Convert AIMessages from the other bot to HumanMessages
                # so Mistral sees them as conversation input
                history.append(HM(content=msg.content, name=msg.name))

        langchain_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_content),
                MessagesPlaceholder("messages"),
            ]
        )
        # Link prompt to Langfuse trace
        langchain_prompt.metadata = {"langfuse_prompt": lf_prompt}

        # 4. Build LLM from prompt config
        model_config = lf_prompt.config or {}
        llm = ChatMistralAI(
            model=model_config.get("model", "mistral-small-latest"),
            temperature=model_config.get("temperature", 0.9),
            api_key=_settings.mistral_api_key,
        )

        # 5. Invoke chain
        chain = langchain_prompt | llm
        response = chain.invoke(
            {"messages": history},
            config=config,
        )
        response.name = state[name_key]

        return {
            "messages": [response],
            turns_key: 1,
        }

    node_fn.__name__ = role  # for LangGraph node naming
    return node_fn


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
initiator = _build_node_fn("initiator", "interview/initiator")
responder = _build_node_fn("responder", "interview/responder")


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------
def after_initiator(state: InterviewState) -> str:
    """Route after initiator speaks: continue or end."""
    if state["responder_turns"] < state["max_turns"]:
        return "responder"
    return END


def after_responder(state: InterviewState) -> str:
    """Route after responder speaks: continue or end."""
    if state["initiator_turns"] < state["max_turns"]:
        return "initiator"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def create_graph(input_schema: type = InputState) -> CompiledStateGraph:
    """Create the two-bot interview graph."""
    builder = StateGraph(InterviewState, input=input_schema)
    builder.add_node("initiator", initiator)
    builder.add_node("responder", responder)
    builder.add_edge(START, "initiator")
    builder.add_conditional_edges("initiator", after_initiator, ["responder", END])
    builder.add_conditional_edges("responder", after_responder, ["initiator", END])
    return builder.compile()


# Export compiled graph for LangGraph Platform
graph = create_graph()

# Register flush on shutdown
atexit.register(get_client().flush)
