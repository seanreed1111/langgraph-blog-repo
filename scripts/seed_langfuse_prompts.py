"""One-time script to create interview prompts in Langfuse.

Run from project root:
    uv run --package stage-2 python scripts/seed_langfuse_prompts.py

This creates two chat prompts with the 'production' label.
If prompts already exist, Langfuse will create a new version.
"""

from langfuse import Langfuse
from stage_2.config import get_settings

PROMPT_TEMPLATE = (
    "You are {{persona_name}}, {{persona_description}}. "
    "{{persona_behavior}} "
    "Keep responses to 2-3 sentences. Do not break character. "
    "Address {{other_persona}} directly."
)

PROMPT_CONFIG = {"model": "mistral-small-latest", "temperature": 0.9}

PROMPTS = [
    "interview/initiator",
    "interview/responder",
]


def main() -> None:
    settings = get_settings()
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
    )

    for name in PROMPTS:
        langfuse.create_prompt(
            name=name,
            type="chat",
            prompt=[{"role": "system", "content": PROMPT_TEMPLATE}],
            config=PROMPT_CONFIG,
            labels=["production"],
        )
        print(f"Created prompt: {name}")

    langfuse.flush()
    print("Done. Prompts created with 'production' label.")


if __name__ == "__main__":
    main()
