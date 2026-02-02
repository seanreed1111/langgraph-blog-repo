# McDonald's Drive-Thru LlamaIndex Workflow Implementation Plan

> **Status:** REVIEWED - See detailed plan for implementation
> **Detailed Implementation:** `plans/2026-02-02-llamaindex-drive-thru-implementation-detailed.md`

## Corrections Applied (2026-02-02)

The following issues were identified during review and corrected in the detailed implementation plan:

| Issue | Original (Incorrect) | Corrected |
|-------|---------------------|-----------|
| LlamaIndex package | `llama-index-core>=0.12.0` | `llama-index>=0.14.0` (unified package) |
| Workflow utils package | `llama-index-utils-workflow>=0.3.0` | Not needed (included in core) |
| Ruff version | `ruff>=0.14.14` | `ruff>=0.9.0` (0.14.x doesn't exist) |
| Missing Anthropic package | N/A | Add `llama-index-llms-anthropic>=0.10.0` |
| Import in models.py | `from enums import ...` | `from src.enums import ...` |
| Context creation in Phase 6 | `Context[OrderState](workflow, store=OrderState())` | Use `Context(workflow)` with `ctx.set("state", state)` |
| Category name mapping | Direct string match | Case-insensitive slug-based mapping (JSON uses "Beef & Pork", enum uses "beef-pork") |

---

> **Status (Original):** DRAFT

## Table of Contents

- [Overview](#overview)
- [Current State Analysis](#current-state-analysis)
- [Desired End State](#desired-end-state)
- [What We're NOT Doing](#what-were-not-doing)
- [Implementation Approach](#implementation-approach)
- [Dependencies](#dependencies)
- [Phase 1: Project Setup and Dependencies](#phase-1-project-setup-and-dependencies)
- [Phase 2: Menu Service and Item Matching](#phase-2-menu-service-and-item-matching)
- [Phase 3: Workflow Events and State](#phase-3-workflow-events-and-state)
- [Phase 4: Tool Definitions](#phase-4-tool-definitions)
- [Phase 5: Workflow Steps](#phase-5-workflow-steps)
- [Phase 6: Workflow Assembly and Entry Point](#phase-6-workflow-assembly-and-entry-point)
- [Phase 7: Testing](#phase-7-testing)
- [Testing Strategy](#testing-strategy)
- [References](#references)

## Overview

Build a McDonald's drive-thru order-taking workflow using **LlamaIndex Workflows** with Anthropic Claude LLMs. The system will:

1. Greet customers as a McDonald's employee
2. Process orders one item at a time through a conversational loop
3. Validate items against a location-specific menu using fuzzy matching
4. Maintain order state across the conversation using LlamaIndex Context
5. Read back the final order and thank the customer

LlamaIndex Workflows provides an event-driven, async-first architecture where:
- **Events** carry data between workflow steps
- **Steps** (decorated with `@step`) handle events and emit new events
- **Context** manages typed state across the workflow
- **Human-in-the-loop** pattern handles the back-and-forth conversation

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

1. **No LlamaIndex integration** - Need workflow, events, state management
2. **No menu loading service** - Need to load JSON menu into searchable format
3. **No item matching logic** - Need strict + fuzzy matching against menu
4. **No conversation management** - Need human-in-the-loop pattern for ordering
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
- [ ] Order state persists across conversation turns via Context
- [ ] Conversation loop continues until customer signals completion
- [ ] Final order summary reads back all items
- [ ] Unit tests cover menu matching and workflow events

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

We'll use **LlamaIndex Workflows** event-driven architecture:

1. **Custom Events** - Define events for each workflow transition (order request, validation result, completion signal)
2. **Typed Context State** - Use Pydantic model with `Context[OrderState]` for type-safe state management
3. **Human-in-the-Loop** - Use `InputRequiredEvent` and `HumanResponseEvent` for conversation
4. **Tool-based Validation** - LLM uses `FunctionTool` to validate items against menu
5. **Async-first** - All workflow steps are async for performance

**Architecture:**

```
StartEvent → greet_step → InputRequiredEvent
                              ↓
                     [Customer Input]
                              ↓
                     HumanResponseEvent
                              ↓
process_order_step → (validate via tools) → OrderUpdatedEvent
                              ↓
                     [check if done]
                              ↓
        [not done] → InputRequiredEvent (loop back)
        [done] → summarize_step → StopEvent
```

## Dependencies

**Execution Order:**

1. Phase 1: Project Setup (no dependencies)
2. Phase 2: Menu Service (no dependencies)
3. Phase 3: Events and State (depends on Phase 2 for menu types)
4. Phase 4: Tool Definitions (depends on Phases 2, 3)
5. Phase 5: Workflow Steps (depends on Phases 3, 4)
6. Phase 6: Workflow Assembly (depends on Phase 5)
7. Phase 7: Testing (depends on Phase 6)

**Dependency Graph:**

```
Phase 1 (Setup)        Phase 2 (Menu Service)
    │                         │
    └──────────┬──────────────┘
               ↓
         Phase 3 (Events/State)
               │
               ↓
         Phase 4 (Tools)
               │
               ↓
         Phase 5 (Steps)
               │
               ↓
         Phase 6 (Assembly)
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
Update dependencies to use LlamaIndex Workflows instead of LangGraph/LangChain. Add fuzzy matching library.

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
Replace langchain/langgraph dependencies with LlamaIndex packages.

```toml
[project]
name = "mcdonalds-data-model"
version = "0.1.0"
description = "McDonald's drive-thru voice ordering data models and workflow"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "llama-index-core>=0.12.0",
    "llama-index-llms-anthropic>=0.6.0",
    "llama-index-utils-workflow>=0.3.0",
    "pydantic>=2.12.5",
    "pydantic-settings>=2.12.0",
    "thefuzz>=0.22.1",
    "python-Levenshtein>=0.26.0",
]

[dependency-groups]
dev = [
    "pre-commit>=4.5.1",
    "ruff>=0.14.14",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "mypy>=1.14.0",
]
```

**Rationale:**
- `llama-index-core` provides the Workflow base class and events
- `llama-index-llms-anthropic` provides the Anthropic Claude integration
- `llama-index-utils-workflow` provides workflow utilities
- `thefuzz` with `python-Levenshtein` provides fast fuzzy string matching

#### 1.2: Create settings module
**File:** `src/settings.py`

**Changes:**
Create settings module for API key and model configuration.

```python
"""Application settings loaded from environment variables."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    anthropic_api_key: str

    # Model configuration
    model_name: str = "claude-sonnet-4-20250514"
    model_temperature: float = 0.3

    # Menu configuration
    menu_path: str = "menus/raw-data/menu-structure-2026-01-30.json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


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
Add MenuItem and MenuItemOption models to represent menu items with their options. Add these after the existing models.

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
                        choices=opt_data.get("choices", []),
                    )

                variations = None
                if "variations" in item_data:
                    var_data = item_data["variations"]
                    variations = MenuItemOption(
                        option_type=var_data.get("type", "addon"),
                        choices=var_data.get("choices", []),
                    )

                menu_item = MenuItem(
                    name=item_data["name"],
                    category=category,
                    available_as_base=item_data.get("available_as_base", True),
                    defaults=item_data.get("defaults", {}),
                    options=options,
                    variations=variations,
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
                score=100,
            )

        # Try fuzzy match
        if not self._item_names:
            return MatchResult(success=False, error="Menu is empty")

        best_match = process.extractOne(
            query, self._item_names, scorer=fuzz.token_sort_ratio
        )

        if best_match and best_match[1] >= self.FUZZY_THRESHOLD:
            matched_name = best_match[0]
            item = self._items[matched_name.lower()]
            return MatchResult(
                success=True,
                menu_item=item,
                matched_name=matched_name,
                score=best_match[1],
            )

        return MatchResult(
            success=False,
            error=f"'{query}' is not on our menu. Please choose from our available items.",
        )

    def list_categories(self) -> list[str]:
        """List all unique menu categories."""
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
- [ ] Menu loads without error: `uv run python -c "from src.menu_service import MenuService; m = MenuService('menus/raw-data/menu-structure-2026-01-30.json'); print(f'Loaded {m.item_count} items')"`
- [ ] Exact match works: Returns success for "Big Mac"
- [ ] Fuzzy match works: Returns success for "bigmac"
- [ ] Invalid item fails: Returns failure for "tacos"

#### Manual Verification:
- [ ] Test various fuzzy inputs manually

---

## Phase 3: Workflow Events and State

### Overview
Define custom events and typed state schema for the LlamaIndex Workflow.

### Context
Before starting, read these files:
- `src/models.py` - Item, Order models
- `src/menu_service.py` - MenuService (from Phase 2)

### Dependencies
**Depends on:** Phase 2
**Required by:** Phase 4, Phase 5

### Changes Required

#### 3.1: Create workflow events
**File:** `src/workflow/events.py`

**Changes:**
Define custom events for workflow transitions.

```python
"""Custom events for the drive-thru ordering workflow."""
from llama_index.core.workflow import Event
from pydantic import Field

from src.models import Item


class CustomerInputEvent(Event):
    """Event carrying customer's spoken input."""

    text: str = Field(description="The customer's input text")


class ValidationRequestEvent(Event):
    """Event requesting validation of an item."""

    item_name: str = Field(description="Name of the item to validate")
    size: str = Field(default="medium", description="Requested size")
    quantity: int = Field(default=1, ge=1, description="Requested quantity")


class ValidationResultEvent(Event):
    """Event carrying the result of item validation."""

    success: bool = Field(description="Whether validation succeeded")
    item: Item | None = Field(default=None, description="Validated item if successful")
    error_message: str | None = Field(default=None, description="Error message if failed")
    original_request: str = Field(description="Original customer request")


class OrderUpdatedEvent(Event):
    """Event indicating the order has been updated."""

    items_added: list[Item] = Field(default_factory=list, description="Items added to order")
    message: str = Field(description="Message to relay to customer")


class OrderCompleteEvent(Event):
    """Event indicating customer is done ordering."""

    pass
```

**Rationale:** Events carry typed data between workflow steps, enabling type-safe transitions.

#### 3.2: Create workflow state
**File:** `src/workflow/state.py`

**Changes:**
Define the typed state schema using Pydantic.

```python
"""Workflow state definitions."""
from pydantic import BaseModel, Field

from src.models import Item


class OrderState(BaseModel):
    """Typed state for the drive-thru ordering workflow.

    All fields must have defaults for LlamaIndex Context compatibility.
    """

    # Conversation state
    is_greeted: bool = Field(default=False, description="Whether customer has been greeted")
    is_complete: bool = Field(default=False, description="Whether ordering is finished")
    turn_count: int = Field(default=0, description="Number of conversation turns")

    # Order state
    current_order: list[Item] = Field(default_factory=list, description="Items in the order")

    # Conversation history (for context to LLM)
    conversation_history: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of {'role': 'user'|'assistant', 'content': str}",
    )

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_order_summary(self) -> str:
        """Generate a summary of the current order."""
        if not self.current_order:
            return "No items in order."

        lines = []
        for item in self.current_order:
            lines.append(f"  - {item.quantity}x {item.name} ({item.size.value})")
        return "\n".join(lines)
```

**Rationale:** Pydantic model provides type safety, validation, and serialization. LlamaIndex Context requires all fields to have defaults.

#### 3.3: Create workflow package init
**File:** `src/workflow/__init__.py`

**Changes:**
```python
"""McDonald's drive-thru ordering workflow using LlamaIndex."""
from src.workflow.events import (
    CustomerInputEvent,
    OrderCompleteEvent,
    OrderUpdatedEvent,
    ValidationRequestEvent,
    ValidationResultEvent,
)
from src.workflow.state import OrderState

__all__ = [
    "CustomerInputEvent",
    "OrderCompleteEvent",
    "OrderUpdatedEvent",
    "OrderState",
    "ValidationRequestEvent",
    "ValidationResultEvent",
]
```

### Success Criteria

#### Automated Verification:
- [ ] Events import correctly: `uv run python -c "from src.workflow.events import CustomerInputEvent, ValidationResultEvent"`
- [ ] State imports correctly: `uv run python -c "from src.workflow.state import OrderState"`
- [ ] State can be instantiated: `uv run python -c "from src.workflow.state import OrderState; s = OrderState(); print(s)"`

#### Manual Verification:
- [ ] State structure makes sense for the workflow

---

## Phase 4: Tool Definitions

### Overview
Define the tools that Claude will use to validate items and manage orders.

### Context
Before starting, read these files:
- `src/workflow/state.py` - OrderState
- `src/menu_service.py` - MenuService, MatchResult
- `src/models.py` - Item model

### Dependencies
**Depends on:** Phase 2, Phase 3
**Required by:** Phase 5

### Changes Required

#### 4.1: Create tools module
**File:** `src/workflow/tools.py`

**Changes:**
Define tools using LlamaIndex's FunctionTool.

```python
"""Tools for the drive-thru ordering workflow."""
from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.tools import FunctionTool

from src.enums import CategoryName, Size
from src.menu_service import MenuService
from src.models import Item

if TYPE_CHECKING:
    pass


def create_tools(menu_service: MenuService) -> list[FunctionTool]:
    """Create workflow tools with the given menu service.

    Args:
        menu_service: The menu service instance to use for validation.

    Returns:
        List of FunctionTool instances for the workflow.
    """

    def validate_menu_item(
        item_name: str,
        size: str = "medium",
        quantity: int = 1,
    ) -> dict:
        """
        Validate if an item exists in the McDonald's menu.

        Use this tool whenever a customer mentions an item they want to order.
        Returns validation result with item details if valid.

        Args:
            item_name: The name of the menu item to validate (e.g., "Big Mac", "fries")
            size: Size of the item: snack, small, medium, or large. Default is medium.
            quantity: Number of this item to order. Must be at least 1. Default is 1.

        Returns:
            Dictionary with validation result including:
            - valid: Whether the item is on the menu
            - item_name: The official menu item name if valid
            - category: The menu category
            - size: The requested size
            - quantity: The requested quantity
            - error: Error message if invalid
        """
        result = menu_service.match_item(item_name)

        if not result.success:
            return {
                "valid": False,
                "error": result.error,
                "suggestion": "Please ask the customer to choose a different item.",
            }

        # Validate size
        try:
            size_enum = Size(size.lower())
        except ValueError:
            return {
                "valid": False,
                "error": f"Invalid size '{size}'. Valid sizes are: snack, small, medium, large.",
                "suggestion": "Ask the customer what size they would like.",
            }

        # Map category string to enum
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

        menu_item = result.menu_item
        category = category_mapping.get(
            menu_item.category if menu_item else "",
            CategoryName.SNACKS_SIDES,
        )

        return {
            "valid": True,
            "item_name": result.matched_name,
            "category": category.value,
            "size": size_enum.value,
            "quantity": quantity,
            "match_score": result.score,
            "message": f"Found '{result.matched_name}' on the menu.",
        }

    def get_menu_categories() -> dict:
        """
        Get the list of available menu categories.

        Use this when a customer asks what's on the menu or what categories are available.

        Returns:
            Dictionary with list of category names.
        """
        return {"categories": menu_service.list_categories()}

    def get_items_in_category(category: str) -> dict:
        """
        Get all items in a specific menu category.

        Use this when a customer asks what items are in a category
        (e.g., "what breakfast items do you have?").

        Args:
            category: The category name to list items from (e.g., "Breakfast", "Beverages")

        Returns:
            Dictionary with list of items in the category, or error if category not found.
        """
        items = menu_service.list_items_in_category(category)

        if not items:
            return {
                "error": f"Category '{category}' not found.",
                "available_categories": menu_service.list_categories(),
            }

        return {
            "category": category,
            "items": items,
            "count": len(items),
        }

    # Create FunctionTool instances
    return [
        FunctionTool.from_defaults(fn=validate_menu_item),
        FunctionTool.from_defaults(fn=get_menu_categories),
        FunctionTool.from_defaults(fn=get_items_in_category),
    ]
```

**Rationale:** Tools are created as closures that capture the menu_service, avoiding global state. FunctionTool.from_defaults infers the schema from type hints and docstrings.

### Success Criteria

#### Automated Verification:
- [ ] Tools module imports: `uv run python -c "from src.workflow.tools import create_tools"`
- [ ] Tools can be created: `uv run python -c "from src.workflow.tools import create_tools; from src.menu_service import MenuService; m = MenuService('menus/raw-data/menu-structure-2026-01-30.json'); tools = create_tools(m); print(f'Created {len(tools)} tools')"`

#### Manual Verification:
- [ ] Tool docstrings are clear for LLM understanding

---

## Phase 5: Workflow Steps

### Overview
Implement the workflow steps that process events and control conversation flow.

### Context
Before starting, read these files:
- `src/workflow/state.py` - OrderState
- `src/workflow/events.py` - Custom events
- `src/workflow/tools.py` - Tool definitions
- `src/settings.py` - Settings

### Dependencies
**Depends on:** Phase 3, Phase 4
**Required by:** Phase 6

### Changes Required

#### 5.1: Create workflow class
**File:** `src/workflow/drive_thru.py`

**Changes:**
Implement the main workflow class with all steps.

```python
"""Drive-thru ordering workflow implementation."""
from __future__ import annotations

import os
from typing import Union

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core.workflow.events import (
    HumanResponseEvent,
    InputRequiredEvent,
)
from llama_index.llms.anthropic import Anthropic

from src.enums import CategoryName, Size
from src.menu_service import MenuService
from src.models import Item
from src.settings import settings
from src.workflow.state import OrderState
from src.workflow.tools import create_tools


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
- Always confirm what size they want if it's not specified (default to medium)
- When they say "that's all", "I'm done", "nothing else", etc., summarize their order

IMPORTANT: Always use the validate_menu_item tool to check items. Do not assume items are valid.
"""


class DriveThruWorkflow(Workflow):
    """Event-driven workflow for McDonald's drive-thru ordering."""

    def __init__(
        self,
        menu_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize menu service
        self.menu_service = MenuService(menu_path or settings.menu_path)

        # Initialize tools
        self.tools = create_tools(self.menu_service)

        # Initialize LLM with tools
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        self.llm = Anthropic(
            model=settings.model_name,
            temperature=settings.model_temperature,
        )

    @step
    async def start(
        self, ctx: Context[OrderState], ev: StartEvent
    ) -> InputRequiredEvent:
        """Initialize workflow and greet the customer."""
        # Initialize state
        async with ctx.store.edit_state() as state:
            state.is_greeted = True
            state.turn_count = 0
            state.current_order = []
            state.conversation_history = []

        greeting = "Welcome to McDonald's! How can I help you today?"

        # Store greeting in conversation history
        async with ctx.store.edit_state() as state:
            state.add_message("assistant", greeting)

        return InputRequiredEvent(prefix=f"McDonald's: {greeting}\n\nYou: ")

    @step
    async def process_input(
        self, ctx: Context[OrderState], ev: HumanResponseEvent
    ) -> Union[InputRequiredEvent, StopEvent]:
        """Process customer input and handle the ordering conversation."""
        customer_input = ev.response.strip()

        if not customer_input:
            return InputRequiredEvent(prefix="You: ")

        # Update state
        async with ctx.store.edit_state() as state:
            state.turn_count += 1
            state.add_message("user", customer_input)
            conversation_history = state.conversation_history.copy()
            current_order = state.current_order.copy()

        # Check for completion signals
        done_phrases = [
            "that's all", "that's it", "i'm done", "im done",
            "nothing else", "no thanks", "nope", "that'll be all",
            "that will be all", "finish", "complete", "checkout",
        ]

        if any(phrase in customer_input.lower() for phrase in done_phrases):
            # Generate order summary
            return await self._generate_summary(ctx, current_order)

        # Safety limit on turns
        if len(conversation_history) > 40:  # 20 back-and-forth exchanges
            return await self._generate_summary(ctx, current_order)

        # Build messages for LLM
        messages = [ChatMessage(role="system", content=ORDER_AGENT_PROMPT)]
        for msg in conversation_history:
            messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

        # Call LLM with tools
        response = await self.llm.achat_with_tools(
            tools=self.tools,
            chat_history=messages,
        )

        # Process tool calls if any
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if tool_calls:
            # Execute tool calls
            for tool_call in tool_calls:
                tool_name = tool_call.tool_name
                tool_kwargs = tool_call.tool_kwargs

                # Find and execute the tool
                for tool in self.tools:
                    if tool.metadata.name == tool_name:
                        tool_result = tool(**tool_kwargs)

                        # If validation succeeded, add item to order
                        if (
                            tool_name == "validate_menu_item"
                            and isinstance(tool_result, dict)
                            and tool_result.get("valid")
                        ):
                            # Create Item from validation result
                            item = Item(
                                item_id=f"item_{len(current_order) + 1}",
                                name=tool_result["item_name"],
                                category_name=CategoryName(tool_result["category"]),
                                size=Size(tool_result["size"]),
                                quantity=tool_result["quantity"],
                            )
                            current_order.append(item)

                            # Update state with new item
                            async with ctx.store.edit_state() as state:
                                state.current_order = current_order

            # Get follow-up response from LLM
            messages.append(
                ChatMessage(role="assistant", content=str(response.message.content))
            )
            messages.append(
                ChatMessage(role="tool", content=str(tool_result))
            )

            follow_up = await self.llm.achat(messages)
            agent_response = str(follow_up.message.content)
        else:
            agent_response = str(response.message.content)

        # Store response in conversation history
        async with ctx.store.edit_state() as state:
            state.add_message("assistant", agent_response)

        # Continue the conversation
        order_status = f"[Order: {len(current_order)} item(s)]" if current_order else ""
        return InputRequiredEvent(
            prefix=f"McDonald's: {agent_response}\n{order_status}\n\nYou: "
        )

    async def _generate_summary(
        self, ctx: Context[OrderState], order: list[Item]
    ) -> StopEvent:
        """Generate the final order summary."""
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
                f"Let me read back your order:\n{items_text}\n\n"
                "Thank you for choosing McDonald's! "
                "Please pull forward to the first window."
            )

        # Update state
        async with ctx.store.edit_state() as state:
            state.is_complete = True
            state.add_message("assistant", summary)

        return StopEvent(result=f"McDonald's: {summary}")
```

**Rationale:**
- Uses `@step` decorator for each workflow step
- Uses `Context[OrderState]` for typed state management
- Uses `InputRequiredEvent` and `HumanResponseEvent` for human-in-the-loop
- Uses `achat_with_tools` for async LLM calls with tool support
- Processes tool calls and updates order state accordingly

### Success Criteria

#### Automated Verification:
- [ ] Workflow imports: `uv run python -c "from src.workflow.drive_thru import DriveThruWorkflow"`
- [ ] Workflow can be instantiated (requires API key): Test in integration tests

#### Manual Verification:
- [ ] Review system prompt for completeness
- [ ] Review tool call handling logic

---

## Phase 6: Workflow Assembly and Entry Point

### Overview
Create the main entry point for running the workflow interactively.

### Context
Before starting, read these files:
- `src/workflow/drive_thru.py` - DriveThruWorkflow class

### Dependencies
**Depends on:** Phase 5
**Required by:** Phase 7

### Changes Required

#### 6.1: Update workflow package init
**File:** `src/workflow/__init__.py`

**Changes:**
Export the main workflow class.

```python
"""McDonald's drive-thru ordering workflow using LlamaIndex."""
from src.workflow.drive_thru import DriveThruWorkflow
from src.workflow.events import (
    CustomerInputEvent,
    OrderCompleteEvent,
    OrderUpdatedEvent,
    ValidationRequestEvent,
    ValidationResultEvent,
)
from src.workflow.state import OrderState

__all__ = [
    "CustomerInputEvent",
    "DriveThruWorkflow",
    "OrderCompleteEvent",
    "OrderUpdatedEvent",
    "OrderState",
    "ValidationRequestEvent",
    "ValidationResultEvent",
]
```

#### 6.2: Create main entry point
**File:** `src/workflow/__main__.py`

**Changes:**
Create interactive CLI for testing the workflow.

```python
"""Interactive CLI for testing the drive-thru workflow."""
from __future__ import annotations

import asyncio

from llama_index.core.workflow import Context
from llama_index.core.workflow.events import HumanResponseEvent

from src.workflow.drive_thru import DriveThruWorkflow
from src.workflow.state import OrderState


async def main():
    """Run interactive drive-thru ordering session."""
    print("=" * 50)
    print("McDonald's Drive-Thru Ordering System")
    print("=" * 50)
    print("Type 'quit' to exit\n")

    # Create workflow with timeout
    workflow = DriveThruWorkflow(timeout=300, verbose=False)

    # Create context with typed state
    ctx = Context[OrderState](workflow, store=OrderState())

    # Start the workflow
    handler = workflow.run(ctx=ctx)

    # Process events
    async for event in handler.stream_events():
        if hasattr(event, "prefix"):
            # InputRequiredEvent - need user input
            print(event.prefix, end="")

            try:
                user_input = input().strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                await handler.cancel_run()
                return

            if user_input.lower() == "quit":
                print("\nGoodbye!")
                await handler.cancel_run()
                return

            # Send user response back to workflow
            handler.ctx.send_event(HumanResponseEvent(response=user_input))

    # Get final result
    try:
        result = await handler
        print(f"\n{result}")
    except Exception as e:
        print(f"\nWorkflow ended: {e}")


def run():
    """Entry point for the CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
```

**Rationale:**
- Uses async event streaming for the conversation loop
- Handles `InputRequiredEvent` by getting user input
- Sends `HumanResponseEvent` back to continue the workflow
- Properly handles cancellation and keyboard interrupts

#### 6.3: Add console script to pyproject.toml
**File:** `pyproject.toml`

**Changes:**
Add after the `[dependency-groups]` section:

```toml
[project.scripts]
drive-thru = "src.workflow.__main__:run"
```

**Rationale:** Allows running the workflow with `uv run drive-thru` command.

### Success Criteria

#### Automated Verification:
- [ ] Entry point imports: `uv run python -c "from src.workflow.__main__ import main"`
- [ ] Script entry point works: `uv run drive-thru` (requires manual test)

#### Manual Verification:
- [ ] Interactive session works: Run and test ordering flow
- [ ] Quit command works
- [ ] Ctrl+C handling works

---

## Phase 7: Testing

### Overview
Write tests for the menu service, events, state, and workflow components.

### Context
Before starting, read these files:
- `src/menu_service.py` - MenuService
- `src/workflow/` - All workflow modules

### Dependencies
**Depends on:** Phase 6
**Required by:** None (final phase)

### Changes Required

#### 7.1: Create test directory structure
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

#### 7.2: Create menu service tests
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
            score=100,
        )
        assert result.success
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = MatchResult(
            success=False,
            error="Item not found",
        )
        assert not result.success
        assert result.matched_name is None
```

#### 7.3: Create workflow state tests
**File:** `tests/test_workflow_state.py`

**Changes:**
```python
"""Tests for workflow state handling."""
import pytest

from src.models import Item
from src.enums import Size, CategoryName
from src.workflow.state import OrderState


class TestOrderState:
    """Tests for OrderState."""

    def test_initial_state(self):
        """Test creating initial state."""
        state = OrderState()
        assert state.is_greeted is False
        assert state.is_complete is False
        assert state.turn_count == 0
        assert len(state.current_order) == 0
        assert len(state.conversation_history) == 0

    def test_add_message(self):
        """Test adding messages to conversation history."""
        state = OrderState()
        state.add_message("assistant", "Welcome!")
        state.add_message("user", "I want a Big Mac")

        assert len(state.conversation_history) == 2
        assert state.conversation_history[0]["role"] == "assistant"
        assert state.conversation_history[1]["content"] == "I want a Big Mac"

    def test_state_with_order(self):
        """Test state with items in order."""
        item = Item(
            item_id="item_1",
            name="Big Mac",
            category_name=CategoryName.BEEF_PORK,
            size=Size.MEDIUM,
            quantity=1,
        )
        state = OrderState(current_order=[item])

        assert len(state.current_order) == 1
        assert state.current_order[0].name == "Big Mac"

    def test_get_order_summary_empty(self):
        """Test order summary with no items."""
        state = OrderState()
        summary = state.get_order_summary()
        assert summary == "No items in order."

    def test_get_order_summary_with_items(self):
        """Test order summary with items."""
        items = [
            Item(
                item_id="item_1",
                name="Big Mac",
                category_name=CategoryName.BEEF_PORK,
                size=Size.MEDIUM,
                quantity=1,
            ),
            Item(
                item_id="item_2",
                name="Large French Fries",
                category_name=CategoryName.SNACKS_SIDES,
                size=Size.LARGE,
                quantity=2,
            ),
        ]
        state = OrderState(current_order=items)
        summary = state.get_order_summary()

        assert "Big Mac" in summary
        assert "Large French Fries" in summary
        assert "2x" in summary
```

#### 7.4: Create tools tests
**File:** `tests/test_workflow_tools.py`

**Changes:**
```python
"""Tests for workflow tools."""
import pytest

from src.menu_service import MenuService
from src.workflow.tools import create_tools


@pytest.fixture
def tools(menu_service):
    """Create tools for testing."""
    return create_tools(menu_service)


@pytest.fixture
def validate_tool(tools):
    """Get the validate_menu_item tool."""
    for tool in tools:
        if tool.metadata.name == "validate_menu_item":
            return tool
    pytest.fail("validate_menu_item tool not found")


@pytest.fixture
def categories_tool(tools):
    """Get the get_menu_categories tool."""
    for tool in tools:
        if tool.metadata.name == "get_menu_categories":
            return tool
    pytest.fail("get_menu_categories tool not found")


@pytest.fixture
def items_tool(tools):
    """Get the get_items_in_category tool."""
    for tool in tools:
        if tool.metadata.name == "get_items_in_category":
            return tool
    pytest.fail("get_items_in_category tool not found")


class TestValidateMenuItem:
    """Tests for validate_menu_item tool."""

    def test_valid_item(self, validate_tool):
        """Test validating a valid menu item."""
        result = validate_tool(
            item_name="Big Mac",
            size="medium",
            quantity=1,
        )
        assert result["valid"] is True
        assert result["item_name"] == "Big Mac"

    def test_valid_item_with_size(self, validate_tool):
        """Test validating item with specific size."""
        result = validate_tool(
            item_name="Coca-Cola Classic",
            size="large",
            quantity=1,
        )
        assert result["valid"] is True
        assert result["size"] == "large"

    def test_invalid_item(self, validate_tool):
        """Test validating an invalid item."""
        result = validate_tool(
            item_name="tacos",
            size="medium",
            quantity=1,
        )
        assert result["valid"] is False
        assert "error" in result

    def test_invalid_size(self, validate_tool):
        """Test validating with invalid size."""
        result = validate_tool(
            item_name="Big Mac",
            size="extra-jumbo",
            quantity=1,
        )
        assert result["valid"] is False
        assert "Invalid size" in result["error"]

    def test_fuzzy_match(self, validate_tool):
        """Test fuzzy matching works through tool."""
        result = validate_tool(
            item_name="mcnuggets",
            size="medium",
            quantity=1,
        )
        assert result["valid"] is True
        assert "Chicken McNuggets" in result["item_name"]


class TestGetMenuCategories:
    """Tests for get_menu_categories tool."""

    def test_returns_categories(self, categories_tool):
        """Test that categories are returned."""
        result = categories_tool()
        assert "categories" in result
        assert len(result["categories"]) > 0


class TestGetItemsInCategory:
    """Tests for get_items_in_category tool."""

    def test_valid_category(self, items_tool):
        """Test getting items from valid category."""
        result = items_tool(category="Breakfast")
        assert "items" in result
        assert len(result["items"]) > 0

    def test_invalid_category(self, items_tool):
        """Test getting items from invalid category."""
        result = items_tool(category="Nonexistent")
        assert "error" in result
```

#### 7.5: Create pytest configuration
**File:** `pyproject.toml`

**Changes:**
Add after `[tool.ruff.format]` section:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
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
- Events: event creation and data passing

### Integration Tests:
- Workflow steps: event handling, state updates
- End-to-end: complete ordering conversation (requires API key)

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
- LlamaIndex Context is in-memory (can be serialized for persistence)
- Async workflow allows non-blocking LLM calls
- Consider adding max token limits for very long conversations

## Migration Notes

N/A - This is a new feature build using LlamaIndex Workflows instead of the previously planned LangGraph implementation.

## References

- Workflow thoughts: `thoughts/workflow-thoughts.md`
- Original LangGraph plan (not implemented): `plans/2026-02-02-langgraph-drive-thru-workflow.md`
- Existing models: `src/models.py`, `src/enums.py`
- Menu data: `menus/raw-data/menu-structure-2026-01-30.json`
- LlamaIndex Workflows docs: https://developers.llamaindex.ai/python/llamaagents/workflows/
- LlamaIndex Anthropic: https://developers.llamaindex.ai/python/examples/llm/anthropic/
- LlamaIndex State Management: https://developers.llamaindex.ai/python/llamaagents/workflows/managing_state/
- LlamaIndex Human-in-the-Loop: https://developers.llamaindex.ai/python/llamaagents/workflows/human_in_the_loop/
