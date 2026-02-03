# Menu Model with JSON Loading Implementation Plan

> **Status:** DRAFT

## Table of Contents

- [Overview](#overview)
- [Current State Analysis](#current-state-analysis)
- [Desired End State](#desired-end-state)
- [What We're NOT Doing](#what-were-not-doing)
- [Implementation Approach](#implementation-approach)
- [Dependencies](#dependencies)
- [Phase 1: Update Models](#phase-1-update-models)
- [Testing Strategy](#testing-strategy)
- [References](#references)

## Overview

Add a `Location` model and update the `Menu` model to support the full JSON structure from `breakfast-v2.json`. The Menu model will have flattened metadata fields and class methods to load from JSON files or dicts.

## Current State Analysis

### Existing Models (`src/models.py`):
- `Modifier` - modifier_id, name
- `Item` - item_id, name, category_name, size, default_size, quantity, modifiers, available_modifiers
- `Order` - order_id, items
- `Menu` - menu_id, name, items (basic, doesn't match JSON structure)

### JSON Structure (`breakfast-v2.json`):
```json
{
  "metadata": {
    "menu_id": "mcd-breakfast-menu",
    "menu_name": "McDonald's Breakfast Menu",
    "menu_version": "v2",
    "location": {
      "id": "mcd-main-location",
      "name": "McDonald's Main Location",
      "address": "123 Main St, Anytown, USA",
      "city": "Anytown",
      "state": "CA",
      "zip": "12345",
      "country": "USA"
    }
  },
  "items": [...]
}
```

### Key Gap:
The current `Menu` model lacks metadata fields (version, location) and loading methods.

## Desired End State

A `Menu` model that:
1. Has flattened metadata fields: `menu_id`, `menu_name`, `menu_version`, `location`
2. Contains a `Location` model for location data
3. Can load from a JSON file path via `Menu.from_json_file(path)`
4. Can load from a dict via `Menu.from_dict(data)`

**Success Criteria:**
- [ ] `Location` model exists with all location fields
- [ ] `Menu` model has flattened metadata fields
- [ ] `Menu.from_json_file()` loads `breakfast-v2.json` successfully
- [ ] `Menu.from_dict()` loads menu data from a dict
- [ ] All items are parsed as `Item` objects with correct types

## What We're NOT Doing

- Adding validation for location fields (address format, zip code, etc.)
- Creating a menu registry or caching mechanism
- Adding write/save functionality
- Modifying the JSON file format

## Implementation Approach

Single-phase update to `src/models.py`:
1. Add `Location` model
2. Update `Menu` model with flattened fields
3. Add `from_dict()` and `from_json_file()` class methods

## Dependencies

**Execution Order:**
1. Phase 1 (no dependencies) - single phase

---

## Phase 1: Update Models

### Overview
Add Location model and update Menu model with loading capabilities.

### Context
Before starting, read these files:
- `src/models.py` - existing models
- `src/enums.py` - existing enums
- `menus/mcdonalds/breakfast-menu/breakfast-v2.json` - target JSON structure

### Dependencies
**Depends on:** None
**Required by:** None

### Changes Required

#### 1.1: Add Location Model
**File:** `src/models.py`

**Changes:**
Add `Location` model after `Modifier` class:

```python
class Location(BaseModel):
    id: str
    name: str
    address: str
    city: str
    state: str
    zip: str
    country: str
```

#### 1.2: Update Menu Model
**File:** `src/models.py`

**Changes:**
Replace existing `Menu` class with updated version:

```python
import json
from pathlib import Path

class Menu(BaseModel):
    menu_id: str
    menu_name: str
    menu_version: str
    location: Location
    items: list[Item]

    @classmethod
    def from_dict(cls, data: dict) -> "Menu":
        """Load Menu from a dictionary (matching JSON structure)."""
        metadata = data["metadata"]
        return cls(
            menu_id=metadata["menu_id"],
            menu_name=metadata["menu_name"],
            menu_version=metadata["menu_version"],
            location=Location(**metadata["location"]),
            items=[Item(**item) for item in data["items"]],
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Menu":
        """Load Menu from a JSON file path."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
```

#### 1.3: Add Required Imports
**File:** `src/models.py`

**Changes:**
Add to imports at top of file:

```python
import json
from pathlib import Path
```

### Success Criteria

#### Automated Verification:
- [ ] Python imports successfully: `uv run python -c "from src.models import Menu, Location"`
- [ ] Load from file works: `uv run python -c "from src.models import Menu; m = Menu.from_json_file('menus/mcdonalds/breakfast-menu/breakfast-v2.json'); print(f'{m.menu_name}: {len(m.items)} items')"`
- [ ] Type checking passes (if mypy configured)

#### Manual Verification:
- [ ] Menu object has correct `menu_id`, `menu_name`, `menu_version`
- [ ] Location object has all fields populated
- [ ] All items are properly parsed with correct category_name enum values

---

## Testing Strategy

### Quick Verification Script:
```python
from src.models import Menu

# Test from_json_file
menu = Menu.from_json_file("menus/mcdonalds/breakfast-menu/breakfast-v2.json")
print(f"Menu: {menu.menu_name}")
print(f"Version: {menu.menu_version}")
print(f"Location: {menu.location.name}, {menu.location.city}, {menu.location.state}")
print(f"Items: {len(menu.items)}")
print(f"First item: {menu.items[0].name}")

# Test from_dict
import json
with open("menus/mcdonalds/breakfast-menu/breakfast-v2.json") as f:
    data = json.load(f)
menu2 = Menu.from_dict(data)
assert menu2.menu_id == menu.menu_id
print("from_dict works!")
```

## References

- JSON source: `menus/mcdonalds/breakfast-menu/breakfast-v2.json`
- Target file: `src/models.py`
