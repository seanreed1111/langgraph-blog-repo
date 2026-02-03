# McDonald's Drive-Thru LangGraph Workflow Implementation Plan

> **Status:** DRAFT

## Table of Contents

- [Overview](#overview)
- [Current State Analysis](#current-state-analysis)
- [Desired End State](#desired-end-state)
- [What We're NOT Doing](#what-were-not-doing)
- [Implementation Approach](#implementation-approach)
- [Dependencies](#dependencies)
- [Phase 1: Project Setup and Dependencies](#phase-1-project-setup-and-dependencies)
- [Phase 2: Menu Service and Item Matching](#phase-2-menu-service-and-item-matching)
- [Phase 3: LangGraph State and Schema](#phase-3-langgraph-state-and-schema)
- [Phase 4: Tool Definitions](#phase-4-tool-definitions)
- [Phase 5: Graph Nodes](#phase-5-graph-nodes)
- [Phase 6: Workflow Graph Assembly](#phase-6-workflow-graph-assembly)
- [Phase 7: Integration and Testing](#phase-7-integration-and-testing)
- [Testing Strategy](#testing-strategy)
- [References](#references)

## Overview

Build a McDonald's drive-thru order-taking workflow using LangGraph and LangChain with Anthropic Claude LLMs. The system will:

1. Greet customers as a McDonald's employee
2. Process orders one item at a time through a conversational loop
3. Validate items against a location-specific menu using fuzzy matching
4. Maintain order state across the conversation
5. Read back the final order and thank the customer

The workflow uses structured Pydantic models for Items and Orders, ensuring type-safe communication between the LLM agent and the order system.

## Current State Analysis

### Existing Code

**`src/enums.py`** - Defines core enums:
- `Size`: snack, small, medium, large
- `CategoryName`: breakfast, beef-pork, chicken-fish, salads, snacks-sides, desserts, beverages, coffee-tea, smoothies-shakes

**`src/models.py`** - Pydantic models:
- `Modifier`: modifier_id, name, allowed_categories
- `Item`: item_id, name, category_name, size (default medium), quantity (default 1), modifiers
- `Order`: order_id (random 1-1000), items list

**Menu Data** - `menus/raw-data/menu-structure-2026-01-30.json`:
- 101 consolidated menu items across 9 categories
- Items have options (size, flavor, preparation style)
- Some items have defaults and variations

### What's Missing

1. **No LangGraph/LangChain integration** - Need workflow, state management, tool definitions
2. **No menu loading service** - Need to load JSON menu into searchable format
3. **No item matching logic** - Need strict + fuzzy matching against menu
4. **No conversation management** - Need greeting, ordering loop, confirmation flow
5. **No Anthropic LLM configuration** - Need Claude model setup with tools

## Desired End State

A working drive-thru ordering system where:

```
Agent: "Welcome to McDonald's! How can I help you today?"
Customer: "I want a Big Mac"
Agent: [validates against menu] "Got it, one Big Mac. Anything else?"
Customer: "Large fries and a medium Coke"
Agent: [validates both items] "Added large fries and medium Coke. Anything else?"
Customer: "That's all"
Agent: "Let me read back your order:
        - 1x Big Mac
        - 1x Large French Fries
        - 1x Medium Coca-Cola Classic
        Thank you for choosing McDonald's!"
```

**Success Criteria:**
- [ ] Menu loads from JSON and supports item lookup
- [ ] Claude model correctly parses customer intent and extracts items
- [ ] Items validate against menu with strict + fuzzy matching
- [ ] Invalid items (like "tacos") are rejected with helpful messages
- [ ] Order state persists across conversation turns
- [ ] Conversation loop continues until customer signals completion
- [ ] Final order summary reads back all items
- [ ] Unit tests cover menu matching and workflow states

**How to Verify:**
```bash
# Run the workflow interactively
uv run python -m src.workflow

# Run tests
uv run pytest tests/ -v

# Type check
uv run mypy src/
```

## What We're NOT Doing

1. **Voice/Speech Integration** - Text-only for now (can add LiveKit later)
2. **Payment Processing** - Order capture only, no payment
3. **Real Database** - In-memory state, no persistence
4. **Multi-location Support** - Single menu file for now
5. **Combo Meals** - Individual items only (Combo model is commented out)
6. **Price Calculations** - No pricing in current menu data
7. **Order Modifications** - No "remove item" or "change size" mid-order
8. **Authentication** - No user accounts or order history

## Implementation Approach

We'll use **LangGraph's StateGraph** pattern with:

1. **TypedDict State** - For performance, storing messages, order, and flow control
2. **Tool-based Validation** - LLM calls `validate_item` and `add_to_order` tools
3. **Conditional Routing** - Route based on customer intent (add item, done ordering)
4. **MemorySaver Checkpointer** - Persist conversation per customer session

**Architecture:**

```
START → greet → order_loop ←→ tools
                    ↓
              [done signal]
                    ↓
               summarize → END
```

## Dependencies

**Execution Order:**

1. Phase 1: Project Setup (no dependencies)
2. Phase 2: Menu Service (no dependencies)
3. Phase 3: State Schema (depends on Phase 2 for Menu type)
4. Phase 4: Tool Definitions (depends on Phases 2, 3)
5. Phase 5: Graph Nodes (depends on Phases 3, 4)
6. Phase 6: Graph Assembly (depends on Phase 5)
7. Phase 7: Integration Testing (depends on Phase 6)

**Dependency Graph:**

```
Phase 1 (Setup)        Phase 2 (Menu Service)
    │                         │
    └──────────┬──────────────┘
               ↓
         Phase 3 (State)
               │
               ↓
         Phase 4 (Tools)
               │
               ↓
         Phase 5 (Nodes)
               │
               ↓
         Phase 6 (Graph)
               │
               ↓
         Phase 7 (Testing)
```

**Parallelization:**
- Phases 1 and 2 can run in parallel (independent)
- Phases 3-7 must run sequentially

---

## Phase 1: Project Setup and Dependencies

### Overview
Add required dependencies and configure Anthropic API access.

### Context
Before starting, read these files:
- `pyproject.toml` - Current dependencies

### Dependencies
**Depends on:** None
**Required by:** All other phases

### Changes Required

#### 1.1: Update pyproject.toml
**File:** `pyproject.toml`

**Changes:**
Add langchain-anthropic and update langgraph version. Add thefuzz for fuzzy matching.

```toml
dependencies = [
    "langchain-anthropic>=0.3.0",
    "langchain-community>=0.4.1",
    "langchain-core>=0.3.0",
    "langgraph>=0.2.0",
    "pydantic>=2.12.5",
    "pydantic-settings>=2.12.0",
    "returns>=0.26.0",
    "thefuzz>=0.22.1",
    "python-Levenshtein>=0.26.0",  # Speed up thefuzz
]
```

**Rationale:** langchain-anthropic provides ChatAnthropic. thefuzz provides fuzzy string matching for menu item lookup.

#### 1.2: Create settings module
**File:** `src/settings.py`

**Changes:**
Create settings module for API key configuration.

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    anthropic_api_key: str

    # Model configuration
    model_name: str = "claude-sonnet-4-20250514"
    model_temperature: float = 0.3

    # Menu configuration
    menu_path: str = "menus/raw-data/menu-structure-2026-01-30.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
```

**Rationale:** Centralized configuration using pydantic-settings with .env file support.

#### 1.3: Create .env.example
**File:** `.env.example`

**Changes:**
```
ANTHROPIC_API_KEY=your-api-key-here
```

**Rationale:** Document required environment variables without exposing secrets.

### Success Criteria

#### Automated Verification:
- [ ] Dependencies install: `uv sync`
- [ ] Settings module imports: `uv run python -c "from src.settings import settings"`

#### Manual Verification:
- [ ] .env file created with valid ANTHROPIC_API_KEY

---

## Phase 2: Menu Service and Item Matching

### Overview
Create a menu service that loads the JSON menu and provides strict + fuzzy item matching.

### Context
Before starting, read these files:
- `menus/raw-data/menu-structure-2026-01-30.json` - Menu structure
- `src/models.py` - Existing Item model
- `src/enums.py` - CategoryName enum

### Dependencies
**Depends on:** None
**Required by:** Phase 3, Phase 4

### Changes Required

#### 2.1: Create MenuItem model
**File:** `src/models.py`

**Changes:**
Add a MenuItem model to represent menu items with their options.

```python
from typing import Any


class MenuItemOption(BaseModel):
    """An option for a menu item (size, flavor, etc.)."""
    option_type: str  # "substitute" or "addon"
    choices: list[str | int]


class MenuItem(BaseModel):
    """A menu item as defined in the restaurant's menu."""
    name: str
    category: str
    available_as_base: bool = True
    defaults: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, MenuItemOption] = Field(default_factory=dict)
    variations: MenuItemOption | None = None
```

**Rationale:** Represents the menu structure from JSON, separate from the order Item model.

#### 2.2: Create menu service
**File:** `src/menu_service.py`

**Changes:**
Create menu loading and item matching service.

```python
"""Menu loading and item matching service."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from thefuzz import fuzz, process

from src.models import MenuItem, MenuItemOption


@dataclass
class MatchResult:
    """Result of a menu item match attempt."""
    success: bool
    menu_item: MenuItem | None = None
    matched_name: str | None = None
    score: int = 0
    error: str | None = None


class MenuService:
    """Service for loading menu and matching customer requests to menu items."""

    # Minimum fuzzy match score to consider a match (0-100)
    FUZZY_THRESHOLD = 80

    def __init__(self, menu_path: str | Path):
        self.menu_path = Path(menu_path)
        self._items: dict[str, MenuItem] = {}
        self._item_names: list[str] = []
        self._load_menu()

    def _load_menu(self) -> None:
        """Load menu from JSON file."""
        with open(self.menu_path) as f:
            data = json.load(f)

        for category, items in data.get("categories", {}).items():
            for item_data in items:
                options = {}
                for opt_name, opt_data in item_data.get("options", {}).items():
                    options[opt_name] = MenuItemOption(
                        option_type=opt_data.get("type", "substitute"),
                        choices=opt_data.get("choices", [])
                    )

                variations = None
                if "variations" in item_data:
                    var_data = item_data["variations"]
                    variations = MenuItemOption(
                        option_type=var_data.get("type", "addon"),
                        choices=var_data.get("choices", [])
                    )

                menu_item = MenuItem(
                    name=item_data["name"],
                    category=category,
                    available_as_base=item_data.get("available_as_base", True),
                    defaults=item_data.get("defaults", {}),
                    options=options,
                    variations=variations
                )

                # Store by lowercase name for case-insensitive lookup
                self._items[item_data["name"].lower()] = menu_item
                self._item_names.append(item_data["name"])

    def get_item(self, name: str) -> MenuItem | None:
        """Get menu item by exact name (case-insensitive)."""
        return self._items.get(name.lower())

    def match_item(self, query: str) -> MatchResult:
        """
        Match a customer query to a menu item.

        Uses strict matching first, then falls back to fuzzy matching.
        Returns MatchResult with success=False if no match found.
        """
        query_lower = query.lower().strip()

        # Try exact match first
        if query_lower in self._items:
            item = self._items[query_lower]
            return MatchResult(
                success=True,
                menu_item=item,
                matched_name=item.name,
                score=100
            )

        # Try fuzzy match
        if not self._item_names:
            return MatchResult(
                success=False,
                error="Menu is empty"
            )

        best_match = process.extractOne(
            query,
            self._item_names,
            scorer=fuzz.token_sort_ratio
        )

        if best_match and best_match[1] >= self.FUZZY_THRESHOLD:
            matched_name = best_match[0]
            item = self._items[matched_name.lower()]
            return MatchResult(
                success=True,
                menu_item=item,
                matched_name=matched_name,
                score=best_match[1]
            )

        return MatchResult(
            success=False,
            error=f"'{query}' is not on our menu. Please choose from our available items."
        )

    def list_categories(self) -> list[str]:
        """List all menu categories."""
        return list(set(item.category for item in self._items.values()))

    def list_items_in_category(self, category: str) -> list[str]:
        """List all item names in a category."""
        return [
            item.name
            for item in self._items.values()
            if item.category.lower() == category.lower()
        ]

    @property
    def item_count(self) -> int:
        """Number of items in the menu."""
        return len(self._items)
```

**Rationale:** Encapsulates menu loading and provides both strict and fuzzy matching. Uses thefuzz's token_sort_ratio which handles word order variations well (e.g., "fries large" matches "Large French Fries").

### Success Criteria

#### Automated Verification:
- [ ] Menu loads without error: `uv run python -c "from src.menu_service import MenuService; m = MenuService('menus/raw-data/menu-structure-2026-01-30.json'); print(m.item_count)"`
- [ ] Exact match works: `m.match_item('Big Mac').success == True`
- [ ] Fuzzy match works: `m.match_item('bigmac').success == True`
- [ ] Invalid item fails: `m.match_item('tacos').success == False`

#### Manual Verification:
- [ ] Test various fuzzy inputs manually

---

## Phase 3: LangGraph State and Schema

### Overview
Define the TypedDict state schema for the LangGraph workflow.

### Context
Before starting, read these files:
- `src/models.py` - Item, Order models
- `src/menu_service.py` - MenuService (from Phase 2)

### Dependencies
**Depends on:** Phase 2
**Required by:** Phase 4, Phase 5

### Changes Required

#### 3.1: Create workflow state
**File:** `src/workflow/state.py`

**Changes:**
Define the state schema using TypedDict with annotated reducers.

```python
"""LangGraph workflow state definitions."""
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.models import Item


class DriveThruState(TypedDict):
    """State for the drive-thru ordering workflow.

    Attributes:
        messages: Conversation history (automatically appended via add_messages)
        current_order: List of validated items in the order
        is_greeted: Whether customer has been greeted
        is_complete: Whether ordering is finished
        turn_count: Number of order turns (for safety limit)
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_order: list[Item]
    is_greeted: bool
    is_complete: bool
    turn_count: int
```

**Rationale:** TypedDict provides performance benefits over Pydantic for state. The `add_messages` annotation automatically handles message list appending.

#### 3.2: Create workflow package init
**File:** `src/workflow/__init__.py`

**Changes:**
```python
"""McDonald's drive-thru ordering workflow."""
from src.workflow.state import DriveThruState

__all__ = ["DriveThruState"]
```

### Success Criteria

#### Automated Verification:
- [ ] State imports correctly: `uv run python -c "from src.workflow.state import DriveThruState"`
- [ ] Type checking passes: `uv run mypy src/workflow/state.py`

#### Manual Verification:
- [ ] State structure makes sense for the workflow

---

## Phase 4: Tool Definitions

### Overview
Define the tools that Claude will use to validate and add items to orders.

### Context
Before starting, read these files:
- `src/workflow/state.py` - DriveThruState
- `src/menu_service.py` - MenuService, MatchResult
- `src/models.py` - Item model

### Dependencies
**Depends on:** Phase 2, Phase 3
**Required by:** Phase 5

### Changes Required

#### 4.1: Create tools module
**File:** `src/workflow/tools.py`

**Changes:**
Define LangChain tools for menu validation and order management.

```python
"""Tools for the drive-thru ordering workflow."""
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from src.enums import Size
from src.menu_service import MenuService
from src.models import Item


# Module-level menu service instance (initialized in graph.py)
_menu_service: MenuService | None = None


def set_menu_service(service: MenuService) -> None:
    """Set the menu service instance for tools to use."""
    global _menu_service
    _menu_service = service


def get_menu_service() -> MenuService:
    """Get the menu service instance."""
    if _menu_service is None:
        raise RuntimeError("Menu service not initialized. Call set_menu_service first.")
    return _menu_service


@tool
def validate_menu_item(
    item_name: Annotated[str, Field(description="The name of the menu item to validate")],
    size: Annotated[str, Field(description="Size: snack, small, medium, or large")] = "medium",
    quantity: Annotated[int, Field(description="Number of this item", ge=1)] = 1,
) -> dict:
    """
    Validate if an item exists in the McDonald's menu.

    Use this tool whenever a customer mentions an item they want to order.
    Returns validation result with item details if valid.
    """
    menu = get_menu_service()
    result = menu.match_item(item_name)

    if not result.success:
        return {
            "valid": False,
            "error": result.error,
            "suggestion": "Please ask the customer to choose a different item."
        }

    # Validate size
    try:
        size_enum = Size(size.lower())
    except ValueError:
        return {
            "valid": False,
            "error": f"Invalid size '{size}'. Valid sizes are: snack, small, medium, large.",
            "suggestion": "Ask the customer what size they would like."
        }

    # Check if item supports the requested size
    menu_item = result.menu_item
    if menu_item and menu_item.options.get("size"):
        valid_sizes = [str(s).lower() for s in menu_item.options["size"].choices]
        if size.lower() not in valid_sizes:
            return {
                "valid": False,
                "error": f"'{result.matched_name}' is not available in {size} size.",
                "valid_sizes": valid_sizes,
                "suggestion": f"Available sizes: {', '.join(valid_sizes)}"
            }

    return {
        "valid": True,
        "item_name": result.matched_name,
        "category": menu_item.category if menu_item else "Unknown",
        "size": size_enum.value,
        "quantity": quantity,
        "match_score": result.score,
        "message": f"Found '{result.matched_name}' on the menu."
    }


@tool
def get_menu_categories() -> dict:
    """
    Get the list of menu categories.

    Use this when a customer asks what's on the menu or what categories are available.
    """
    menu = get_menu_service()
    return {
        "categories": menu.list_categories()
    }


@tool
def get_items_in_category(
    category: Annotated[str, Field(description="The category name to list items from")]
) -> dict:
    """
    Get all items in a specific menu category.

    Use this when a customer asks what items are in a category (e.g., "what breakfast items do you have?").
    """
    menu = get_menu_service()
    items = menu.list_items_in_category(category)

    if not items:
        return {
            "error": f"Category '{category}' not found.",
            "available_categories": menu.list_categories()
        }

    return {
        "category": category,
        "items": items,
        "count": len(items)
    }
```

**Rationale:** Tools use docstrings and type annotations that Claude can understand. The validate_menu_item tool is the primary interface for item validation.

### Success Criteria

#### Automated Verification:
- [ ] Tools import correctly: `uv run python -c "from src.workflow.tools import validate_menu_item, get_menu_categories"`
- [ ] Tool schema is generated: Check tool.args_schema

#### Manual Verification:
- [ ] Tool descriptions are clear for LLM understanding

---

## Phase 5: Graph Nodes

### Overview
Implement the node functions that make up the workflow graph.

### Context
Before starting, read these files:
- `src/workflow/state.py` - DriveThruState
- `src/workflow/tools.py` - Tool definitions
- `src/settings.py` - Settings

### Dependencies
**Depends on:** Phase 3, Phase 4
**Required by:** Phase 6

### Changes Required

#### 5.1: Create nodes module
**File:** `src/workflow/nodes.py`

**Changes:**
Implement the workflow nodes.

```python
"""Node functions for the drive-thru workflow."""
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from src.models import Item
from src.enums import Size, CategoryName
from src.settings import settings
from src.workflow.state import DriveThruState
from src.workflow.tools import (
    validate_menu_item,
    get_menu_categories,
    get_items_in_category,
)


# System prompt for the order-taking agent
ORDER_AGENT_PROMPT = """You are a friendly McDonald's drive-thru employee taking orders.

Your responsibilities:
1. Greet customers warmly when they first arrive
2. Listen to their order and use the validate_menu_item tool to check each item
3. Confirm validated items and ask if they want anything else
4. When they're done, summarize the complete order

Guidelines:
- Be conversational and friendly, like a real drive-thru employee
- Process ONE item at a time - validate it before moving on
- If an item isn't on the menu, politely let them know and suggest alternatives
- If they ask about the menu, use get_menu_categories or get_items_in_category
- Always confirm what size they want if it's not specified
- When they say "that's all" or "I'm done", summarize their order

Remember: You are validating items, not adding them to the order directly.
The system handles order management based on your validated items."""


# Tools available to the agent
TOOLS = [validate_menu_item, get_menu_categories, get_items_in_category]


def get_model():
    """Get the configured Claude model with tools bound."""
    model = ChatAnthropic(
        model=settings.model_name,
        temperature=settings.model_temperature,
        api_key=settings.anthropic_api_key,
    )
    return model.bind_tools(TOOLS)


def greeting_node(state: DriveThruState) -> dict:
    """Initial greeting node - welcomes the customer."""
    greeting = AIMessage(
        content="Welcome to McDonald's! How can I help you today?"
    )
    return {
        "messages": [greeting],
        "is_greeted": True,
        "current_order": [],
        "turn_count": 0,
    }


def agent_node(state: DriveThruState) -> dict:
    """Main agent node - processes customer input and decides actions."""
    model = get_model()

    # Build messages with system prompt
    messages = [SystemMessage(content=ORDER_AGENT_PROMPT)] + list(state["messages"])

    # Get model response
    response = model.invoke(messages)

    return {
        "messages": [response],
        "turn_count": state["turn_count"] + 1,
    }


def tool_node(state: DriveThruState) -> dict:
    """Execute tool calls from the agent."""
    tool_executor = ToolNode(tools=TOOLS)
    return tool_executor.invoke(state)


def process_validation_node(state: DriveThruState) -> dict:
    """Process validation results and update order if valid."""
    messages = state["messages"]
    current_order = list(state["current_order"])

    # Look for recent tool messages with validation results
    for msg in reversed(messages[-5:]):  # Check last 5 messages
        if isinstance(msg, ToolMessage) and msg.name == "validate_menu_item":
            try:
                import json
                result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content

                if isinstance(result, dict) and result.get("valid"):
                    # Create Item from validated data
                    # Map category string to CategoryName enum
                    category_mapping = {
                        "Breakfast": CategoryName.BREAKFAST,
                        "Beef & Pork": CategoryName.BEEF_PORK,
                        "Chicken & Fish": CategoryName.CHICKEN_FISH,
                        "Salads": CategoryName.SALADS,
                        "Snacks & Sides": CategoryName.SNACKS_SIDES,
                        "Desserts": CategoryName.DESSERTS,
                        "Beverages": CategoryName.BEVERAGES,
                        "Coffee & Tea": CategoryName.COFFEE_TEA,
                        "Smoothies & Shakes": CategoryName.SMOOTHIES_SHAKES,
                    }

                    category = category_mapping.get(
                        result.get("category", ""),
                        CategoryName.SNACKS_SIDES
                    )

                    item = Item(
                        item_id=f"item_{len(current_order) + 1}",
                        name=result["item_name"],
                        category_name=category,
                        size=Size(result.get("size", "medium")),
                        quantity=result.get("quantity", 1),
                    )
                    current_order.append(item)
                    break
            except (json.JSONDecodeError, KeyError, ValueError):
                pass  # Skip malformed results

    return {"current_order": current_order}


def summary_node(state: DriveThruState) -> dict:
    """Summarize the order and thank the customer."""
    order = state["current_order"]

    if not order:
        summary = (
            "It looks like you didn't order anything today. "
            "No problem! Come back anytime when you're ready to order. "
            "Thank you for visiting McDonald's!"
        )
    else:
        items_text = "\n".join([
            f"  - {item.quantity}x {item.name} ({item.size.value})"
            for item in order
        ])
        summary = (
            f"Here's your order:\n{items_text}\n\n"
            "Thank you for choosing McDonald's! "
            "Please pull forward to the first window."
        )

    return {
        "messages": [AIMessage(content=summary)],
        "is_complete": True,
    }


# Routing functions

def should_continue(state: DriveThruState) -> Literal["tools", "process", "continue"]:
    """Determine next step after agent node."""
    last_message = state["messages"][-1]

    # If agent wants to use tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "continue"


def check_if_done(state: DriveThruState) -> Literal["summarize", "agent"]:
    """Check if customer is done ordering."""
    # Safety limit on turns
    if state["turn_count"] >= 20:
        return "summarize"

    # Check last message for completion signals
    messages = state["messages"]
    if messages:
        last_content = ""
        for msg in reversed(messages[-3:]):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                last_content = msg.content.lower()
                break

        done_phrases = [
            "that's all", "that's it", "i'm done", "im done",
            "nothing else", "no thanks", "nope", "that'll be all",
            "that will be all", "finish", "complete", "checkout"
        ]

        if any(phrase in last_content for phrase in done_phrases):
            return "summarize"

    return "agent"


def after_tools(state: DriveThruState) -> Literal["process", "agent"]:
    """Route after tool execution."""
    # Check if the last tool call was validate_menu_item
    messages = state["messages"]
    for msg in reversed(messages[-3:]):
        if isinstance(msg, ToolMessage) and msg.name == "validate_menu_item":
            return "process"
    return "agent"
```

**Rationale:** Nodes are pure functions that take state and return state updates. The routing functions determine graph flow based on conversation state.

### Success Criteria

#### Automated Verification:
- [ ] Nodes import correctly: `uv run python -c "from src.workflow.nodes import greeting_node, agent_node"`
- [ ] Model instantiation works (requires API key)

#### Manual Verification:
- [ ] Review system prompt for completeness

---

## Phase 6: Workflow Graph Assembly

### Overview
Assemble the complete LangGraph workflow from nodes and routing logic.

### Context
Before starting, read these files:
- `src/workflow/nodes.py` - Node functions
- `src/workflow/state.py` - State schema
- `src/workflow/tools.py` - Tool setup

### Dependencies
**Depends on:** Phase 5
**Required by:** Phase 7

### Changes Required

#### 6.1: Create graph module
**File:** `src/workflow/graph.py`

**Changes:**
Build the StateGraph workflow.

```python
"""LangGraph workflow assembly for drive-thru ordering."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.menu_service import MenuService
from src.settings import settings
from src.workflow.nodes import (
    agent_node,
    greeting_node,
    process_validation_node,
    summary_node,
    tool_node,
    should_continue,
    check_if_done,
    after_tools,
)
from src.workflow.state import DriveThruState
from src.workflow.tools import set_menu_service


def create_workflow(menu_path: str | None = None) -> StateGraph:
    """
    Create the drive-thru ordering workflow graph.

    Args:
        menu_path: Path to menu JSON file. Uses settings default if not provided.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    # Initialize menu service
    menu_service = MenuService(menu_path or settings.menu_path)
    set_menu_service(menu_service)

    # Create the graph
    workflow = StateGraph(DriveThruState)

    # Add nodes
    workflow.add_node("greet", greeting_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process", process_validation_node)
    workflow.add_node("summarize", summary_node)

    # Add edges
    # Start with greeting
    workflow.add_edge(START, "greet")
    workflow.add_edge("greet", "agent")

    # Agent decides: use tools, or continue conversation
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "continue": "check_done",
        }
    )

    # After tools: process validation or return to agent
    workflow.add_conditional_edges(
        "tools",
        after_tools,
        {
            "process": "process",
            "agent": "agent",
        }
    )

    # After processing validation, check if done
    workflow.add_edge("process", "check_done")

    # Check if customer is done ordering
    workflow.add_node("check_done", lambda state: state)  # Pass-through node for routing
    workflow.add_conditional_edges(
        "check_done",
        check_if_done,
        {
            "summarize": "summarize",
            "agent": "agent",
        }
    )

    # End after summary
    workflow.add_edge("summarize", END)

    return workflow


def create_app(menu_path: str | None = None):
    """
    Create a compiled workflow app with checkpointing.

    Args:
        menu_path: Path to menu JSON file.

    Returns:
        Compiled workflow with memory checkpointer.
    """
    workflow = create_workflow(menu_path)
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
```

**Rationale:** The graph follows the pattern: greet → agent ↔ tools → check_done → agent (loop) or summarize → END.

#### 6.2: Update workflow package init
**File:** `src/workflow/__init__.py`

**Changes:**
Export the main entry points.

```python
"""McDonald's drive-thru ordering workflow."""
from src.workflow.graph import create_app, create_workflow
from src.workflow.state import DriveThruState

__all__ = ["create_app", "create_workflow", "DriveThruState"]
```

#### 6.3: Create main entry point
**File:** `src/workflow/__main__.py`

**Changes:**
Create interactive CLI for testing the workflow.

```python
"""Interactive CLI for testing the drive-thru workflow."""
import uuid

from langchain_core.messages import HumanMessage

from src.workflow import create_app


def main():
    """Run interactive drive-thru ordering session."""
    print("=" * 50)
    print("McDonald's Drive-Thru Ordering System")
    print("=" * 50)
    print("Type 'quit' to exit\n")

    # Create app and unique thread ID for this session
    app = create_app()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state
    initial_state = {
        "messages": [],
        "current_order": [],
        "is_greeted": False,
        "is_complete": False,
        "turn_count": 0,
    }

    # Start the conversation (triggers greeting)
    result = app.invoke(initial_state, config=config)

    # Print greeting
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(f"\nMcDonald's: {msg.content}")

    # Conversation loop
    while not result.get("is_complete", False):
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Send user message and get response
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # Print agent response
            for msg in result["messages"]:
                if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                    print(f"\nMcDonald's: {msg.content}")

            # Print current order status
            if result.get("current_order"):
                print(f"\n[Order has {len(result['current_order'])} item(s)]")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
```

**Rationale:** Provides an interactive way to test the complete workflow.

### Success Criteria

#### Automated Verification:
- [ ] Graph compiles: `uv run python -c "from src.workflow import create_app; app = create_app()"`
- [ ] Graph visualization works (optional): `app.get_graph().draw_mermaid()`

#### Manual Verification:
- [ ] Interactive session works: `uv run python -m src.workflow`

---

## Phase 7: Integration and Testing

### Overview
Write tests for the menu service and workflow components.

### Context
Before starting, read these files:
- `src/menu_service.py` - MenuService
- `src/workflow/` - All workflow modules

### Dependencies
**Depends on:** Phase 6
**Required by:** None (final phase)

### Changes Required

#### 7.1: Add pytest to dev dependencies
**File:** `pyproject.toml`

**Changes:**
Add pytest and pytest-asyncio to dev dependencies.

```toml
[dependency-groups]
dev = [
    "pre-commit>=4.5.1",
    "ruff>=0.14.14",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "mypy>=1.14.0",
]
```

#### 7.2: Create test directory structure
**Files:**
- `tests/__init__.py`
- `tests/conftest.py`

**Changes for conftest.py:**
```python
"""Shared test fixtures."""
import pytest

from src.menu_service import MenuService


@pytest.fixture
def menu_service():
    """Create a MenuService instance for testing."""
    return MenuService("menus/raw-data/menu-structure-2026-01-30.json")


@pytest.fixture
def sample_menu_items():
    """Common menu items for testing."""
    return [
        "Big Mac",
        "Quarter Pounder with Cheese",
        "Chicken McNuggets",
        "Large French Fries",
        "Coca-Cola Classic",
    ]
```

#### 7.3: Create menu service tests
**File:** `tests/test_menu_service.py`

**Changes:**
```python
"""Tests for the menu service."""
import pytest

from src.menu_service import MenuService, MatchResult


class TestMenuService:
    """Tests for MenuService class."""

    def test_load_menu(self, menu_service: MenuService):
        """Test menu loads successfully."""
        assert menu_service.item_count > 0
        assert menu_service.item_count == 101  # From metadata

    def test_exact_match(self, menu_service: MenuService):
        """Test exact item name matching."""
        result = menu_service.match_item("Big Mac")
        assert result.success
        assert result.matched_name == "Big Mac"
        assert result.score == 100

    def test_case_insensitive_match(self, menu_service: MenuService):
        """Test case-insensitive matching."""
        result = menu_service.match_item("big mac")
        assert result.success
        assert result.matched_name == "Big Mac"

    def test_fuzzy_match(self, menu_service: MenuService):
        """Test fuzzy matching for typos."""
        result = menu_service.match_item("bigmac")
        assert result.success
        assert result.matched_name == "Big Mac"
        assert result.score >= 80

    def test_fuzzy_match_partial(self, menu_service: MenuService):
        """Test fuzzy matching for partial names."""
        result = menu_service.match_item("quarter pounder cheese")
        assert result.success
        assert "Quarter Pounder" in result.matched_name

    def test_invalid_item(self, menu_service: MenuService):
        """Test that invalid items return failure."""
        result = menu_service.match_item("tacos")
        assert not result.success
        assert result.error is not None
        assert "not on our menu" in result.error

    def test_invalid_item_sushi(self, menu_service: MenuService):
        """Test another invalid item."""
        result = menu_service.match_item("sushi roll")
        assert not result.success

    def test_list_categories(self, menu_service: MenuService):
        """Test listing menu categories."""
        categories = menu_service.list_categories()
        assert len(categories) == 9
        assert "Breakfast" in categories
        assert "Beef & Pork" in categories

    def test_list_items_in_category(self, menu_service: MenuService):
        """Test listing items in a category."""
        items = menu_service.list_items_in_category("Breakfast")
        assert len(items) > 0
        assert "Egg McMuffin" in items

    def test_list_items_invalid_category(self, menu_service: MenuService):
        """Test listing items for invalid category."""
        items = menu_service.list_items_in_category("Nonexistent")
        assert len(items) == 0


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = MatchResult(
            success=True,
            matched_name="Big Mac",
            score=100
        )
        assert result.success
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = MatchResult(
            success=False,
            error="Item not found"
        )
        assert not result.success
        assert result.matched_name is None
```

#### 7.4: Create workflow state tests
**File:** `tests/test_workflow_state.py`

**Changes:**
```python
"""Tests for workflow state handling."""
import pytest

from langchain_core.messages import AIMessage, HumanMessage

from src.models import Item
from src.enums import Size, CategoryName
from src.workflow.state import DriveThruState


class TestDriveThruState:
    """Tests for DriveThruState."""

    def test_initial_state(self):
        """Test creating initial state."""
        state: DriveThruState = {
            "messages": [],
            "current_order": [],
            "is_greeted": False,
            "is_complete": False,
            "turn_count": 0,
        }
        assert state["is_greeted"] is False
        assert len(state["messages"]) == 0

    def test_state_with_messages(self):
        """Test state with messages."""
        state: DriveThruState = {
            "messages": [
                AIMessage(content="Welcome!"),
                HumanMessage(content="I want a Big Mac"),
            ],
            "current_order": [],
            "is_greeted": True,
            "is_complete": False,
            "turn_count": 1,
        }
        assert len(state["messages"]) == 2
        assert state["is_greeted"] is True

    def test_state_with_order(self):
        """Test state with items in order."""
        item = Item(
            item_id="item_1",
            name="Big Mac",
            category_name=CategoryName.BEEF_PORK,
            size=Size.MEDIUM,
            quantity=1,
        )
        state: DriveThruState = {
            "messages": [],
            "current_order": [item],
            "is_greeted": True,
            "is_complete": False,
            "turn_count": 1,
        }
        assert len(state["current_order"]) == 1
        assert state["current_order"][0].name == "Big Mac"
```

#### 7.5: Create tools tests
**File:** `tests/test_workflow_tools.py`

**Changes:**
```python
"""Tests for workflow tools."""
import pytest

from src.menu_service import MenuService
from src.workflow.tools import (
    validate_menu_item,
    get_menu_categories,
    get_items_in_category,
    set_menu_service,
)


@pytest.fixture(autouse=True)
def setup_menu_service():
    """Set up menu service before each test."""
    service = MenuService("menus/raw-data/menu-structure-2026-01-30.json")
    set_menu_service(service)


class TestValidateMenuItem:
    """Tests for validate_menu_item tool."""

    def test_valid_item(self):
        """Test validating a valid menu item."""
        result = validate_menu_item.invoke({
            "item_name": "Big Mac",
            "size": "medium",
            "quantity": 1
        })
        assert result["valid"] is True
        assert result["item_name"] == "Big Mac"

    def test_valid_item_with_size(self):
        """Test validating item with specific size."""
        result = validate_menu_item.invoke({
            "item_name": "Coca-Cola Classic",
            "size": "large",
            "quantity": 1
        })
        assert result["valid"] is True
        assert result["size"] == "large"

    def test_invalid_item(self):
        """Test validating an invalid item."""
        result = validate_menu_item.invoke({
            "item_name": "tacos",
            "size": "medium",
            "quantity": 1
        })
        assert result["valid"] is False
        assert "error" in result

    def test_invalid_size(self):
        """Test validating with invalid size."""
        result = validate_menu_item.invoke({
            "item_name": "Big Mac",
            "size": "extra-jumbo",
            "quantity": 1
        })
        assert result["valid"] is False
        assert "Invalid size" in result["error"]

    def test_fuzzy_match(self):
        """Test fuzzy matching works through tool."""
        result = validate_menu_item.invoke({
            "item_name": "mcnuggets",
            "size": "medium",
            "quantity": 1
        })
        assert result["valid"] is True
        assert "Chicken McNuggets" in result["item_name"]


class TestGetMenuCategories:
    """Tests for get_menu_categories tool."""

    def test_returns_categories(self):
        """Test that categories are returned."""
        result = get_menu_categories.invoke({})
        assert "categories" in result
        assert len(result["categories"]) > 0


class TestGetItemsInCategory:
    """Tests for get_items_in_category tool."""

    def test_valid_category(self):
        """Test getting items from valid category."""
        result = get_items_in_category.invoke({"category": "Breakfast"})
        assert "items" in result
        assert len(result["items"]) > 0

    def test_invalid_category(self):
        """Test getting items from invalid category."""
        result = get_items_in_category.invoke({"category": "Nonexistent"})
        assert "error" in result
```

### Success Criteria

#### Automated Verification:
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Type checking passes: `uv run mypy src/`
- [ ] Linting passes: `uv run ruff check src/ tests/`

#### Manual Verification:
- [ ] Run interactive workflow and complete an order
- [ ] Test edge cases: invalid items, size validation, order completion

---

## Testing Strategy

### Unit Tests:
- Menu service: item loading, matching, fuzzy search
- State management: state creation and updates
- Tools: validation logic, error handling

### Integration Tests:
- Workflow graph: node transitions, routing logic
- End-to-end: complete ordering conversation

### Manual Testing Steps:
1. Start the workflow: `uv run python -m src.workflow`
2. Test greeting: Verify welcome message appears
3. Test valid order: "I want a Big Mac" → should validate and confirm
4. Test invalid order: "I want tacos" → should reject politely
5. Test fuzzy match: "mcnuggets" → should match "Chicken McNuggets"
6. Test completion: "That's all" → should summarize order
7. Test multiple items: Order 3+ items, verify all in summary

## Performance Considerations

- Menu is loaded once and cached in MenuService
- Fuzzy matching uses thefuzz with python-Levenshtein for speed
- MemorySaver checkpointer is in-memory (use PostgresSaver for production)
- Consider adding max token limits for very long conversations

## Migration Notes

N/A - This is a new feature build.

## References

- Workflow thoughts: `thoughts/workflow-thoughts.md`
- Existing models: `src/models.py`, `src/enums.py`
- Menu data: `menus/raw-data/menu-structure-2026-01-30.json`
- LangGraph docs: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangChain Anthropic: https://docs.langchain.com/oss/python/integrations/chat/anthropic
