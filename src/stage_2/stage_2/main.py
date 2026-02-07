"""CLI for the two-bot interview graph with streaming output."""

import argparse
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from stage_2.graph import graph, get_langfuse_handler, get_langfuse_client
from stage_2.personas import Preset, PERSONA_PRESETS


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Two-Bot Interview")
    parser.add_argument(
        "--preset",
        choices=[p.value for p in Preset],
        default=Preset.REPORTER_POLITICIAN.value,
        help="Persona pairing (default: reporter-politician)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="Max messages per bot (default: 3)",
    )
    return parser.parse_args()


def run_interview() -> None:
    """Run a single interview conversation and print the transcript."""
    args = parse_args()
    preset_key = Preset(args.preset)
    max_turns = args.max_turns
    preset = PERSONA_PRESETS[preset_key]

    initiator_name = preset["initiator"]["persona_name"]
    responder_name = preset["responder"]["persona_name"]

    # Session tracking
    session_id = f"interview-{uuid.uuid4().hex[:8]}"

    print(f"Two-Bot Interview: {initiator_name} vs {responder_name}")
    print(f"Preset: {preset_key} | Max turns: {max_turns} | Session: {session_id}")
    print("=" * 60)

    # Get topic from user
    try:
        topic = input("\nEnter interview topic: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if not topic:
        print("No topic provided. Exiting.")
        return

    print(f"\nTopic: {topic}")
    print("-" * 60)

    # Set up Langfuse tracing
    langfuse_handler = get_langfuse_handler()

    # Invoke graph with streaming (node-by-node)
    try:
        for update in graph.stream(
            {
                "messages": [HumanMessage(content=f"Interview topic: {topic}")],
                "max_turns": max_turns,
                "initiator_turns": 0,
                "responder_turns": 0,
                "preset": preset_key,
                "initiator_name": initiator_name,
                "responder_name": responder_name,
            },
            config={
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_session_id": session_id,
                },
            },
            stream_mode="updates",
        ):
            # update is {"node_name": {"messages": [AIMessage], ...}}
            for node_name, node_output in update.items():
                if "messages" not in node_output:
                    continue
                for msg in node_output["messages"]:
                    if isinstance(msg, AIMessage):
                        speaker = msg.name or node_name
                        print(f"\n[{speaker}]: {msg.content}")
    except Exception as e:
        print(f"\nError during interview: {e}")
        raise

    # Summary
    print("\n" + "=" * 60)
    print("Interview complete.")
    print(f"Total turns: {max_turns} per bot ({max_turns * 2} messages)")

    # Flush traces
    get_langfuse_client().flush()


if __name__ == "__main__":
    run_interview()
