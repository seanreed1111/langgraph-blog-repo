# CSV to JSON Transformation Plan (v2)

> **Status:** DRAFT

## Overview

Transform `menus/transformed-data/mcdonalds-menu-items-revised-full.csv` into a JSON file that validates against the updated Pydantic models in `src/models.py`.

The updated `Menu` model now includes an `items` list, making this a straightforward Option B implementation from the original plan.

## Current State

### CSV Structure
- 269 rows (including duplicates)
- Two columns: `Category`, `Item`
- 9 categories matching `CategoryName` enum
- Size info embedded in item names (e.g., "Coca-Cola Classic, Small")
- Piece counts in some items (e.g., "Chicken McNuggets, 4 pc")
- "Regular"/"Large" variants for some breakfast items

### Updated Pydantic Models (`src/models.py`)

```python
class Size(StrEnum):
    SNACK = "snack"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class CategoryName(StrEnum):
    BREAKFAST = "breakfast"
    BEEF_PORK = "beef-pork"
    CHICKEN_FISH = "chicken-fish"
    SALADS = "salads"
    SNACKS_SIDES = "snacks-sides"
    DESSERTS = "desserts"
    BEVERAGES = "beverages"
    COFFEE_TEA = "coffee-tea"
    SMOOTHIES_SHAKES = "smoothies-shakes"

class Item(BaseModel):
    item_id: str
    name: str
    category_name: CategoryName
    size: Size = Field(default=Size.MEDIUM)
    quantity: int = Field(default=1, ge=1)
    modifiers: list[Modifier] = Field(default_factory=list)

class Menu(BaseModel):
    menu_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    items: list[Item]
```

## Design Decisions

| Decision | Choice |
|----------|--------|
| Size variations | One `Item` per CSV row |
| Item ID format | Slugified from item name |
| Piece counts | Keep in name AND map to size |
| "Regular" size | Maps to `"medium"` |
| Duplicates | Keep first occurrence only |
| "Kids" size | Maps to `"small"` |

### Size Mapping

| CSV Pattern | Size Enum |
|-------------|-----------|
| `, Small` | `small` |
| `, Medium` | `medium` |
| `, Large` | `large` |
| `, Kids` | `small` |
| `, Snack` | `snack` |
| `, Regular` | `medium` |
| `, 4 pc` | `snack` |
| `, 6 pc` | `small` |
| `, 10 pc` | `medium` |
| `, 20 pc` | `large` |
| `, 40 pc` | `large` |
| (no size) | `medium` (default) |

### Category Mapping

| CSV Category | CategoryName Enum |
|--------------|-------------------|
| Breakfast | `breakfast` |
| Beef & Pork | `beef-pork` |
| Chicken & Fish | `chicken-fish` |
| Salads | `salads` |
| Snacks & Sides | `snacks-sides` |
| Desserts | `desserts` |
| Beverages | `beverages` |
| Coffee & Tea | `coffee-tea` |
| Smoothies & Shakes | `smoothies-shakes` |

## Desired End State

A JSON file that validates with:
```python
from src.models import Menu
import json

with open("menus/transformed-data/menu.json") as f:
    menu = Menu.model_validate(json.load(f))
```

### Example Output Structure

```json
{
  "menu_id": "mcd-main-menu",
  "name": "McDonald's Menu",
  "items": [
    {
      "item_id": "egg-mcmuffin",
      "name": "Egg McMuffin",
      "category_name": "breakfast",
      "size": "medium",
      "quantity": 1,
      "modifiers": []
    },
    {
      "item_id": "coca-cola-classic-small",
      "name": "Coca-Cola Classic, Small",
      "category_name": "beverages",
      "size": "small",
      "quantity": 1,
      "modifiers": []
    },
    {
      "item_id": "chicken-mcnuggets-4-pc",
      "name": "Chicken McNuggets, 4 pc",
      "category_name": "chicken-fish",
      "size": "snack",
      "quantity": 1,
      "modifiers": []
    }
  ]
}
```

## What We're NOT Doing

- NOT populating modifiers (CSV has no modifier data)
- NOT creating combos
- NOT adding pricing
- NOT modifying Pydantic models

## Phase 1: Create Transformation Script

### File: `scripts/csv_to_json.py`

```python
"""Transform McDonald's menu CSV to JSON matching Pydantic models."""

import csv
import json
import re
from pathlib import Path

# Import for validation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import Menu
from enums import CategoryName, Size


def slugify(text: str) -> str:
    """Convert text to kebab-case slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


# CSV category name -> CategoryName enum value
CATEGORY_MAP = {
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

# Piece count -> Size mapping
PIECE_COUNT_SIZE_MAP = {
    4: Size.SNACK,
    6: Size.SMALL,
    10: Size.MEDIUM,
    20: Size.LARGE,
    40: Size.LARGE,
}


def extract_size(item_name: str) -> Size:
    """Extract size from item name, return Size enum."""
    name_lower = item_name.lower()

    # Check for explicit size suffixes
    if name_lower.endswith(", small"):
        return Size.SMALL
    if name_lower.endswith(", medium"):
        return Size.MEDIUM
    if name_lower.endswith(", large"):
        return Size.LARGE
    if name_lower.endswith(", kids"):
        return Size.SMALL
    if name_lower.endswith(", snack"):
        return Size.SNACK
    if name_lower.endswith(", regular"):
        return Size.MEDIUM

    # Check for piece count pattern
    pc_match = re.search(r",\s*(\d+)\s*pc$", item_name, re.IGNORECASE)
    if pc_match:
        piece_count = int(pc_match.group(1))
        return PIECE_COUNT_SIZE_MAP.get(piece_count, Size.MEDIUM)

    # Default
    return Size.MEDIUM


def main():
    csv_path = Path("menus/transformed-data/mcdonalds-menu-items-revised-full.csv")
    output_path = Path("menus/transformed-data/menu.json")

    seen_items = set()  # For deduplication
    items = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category_csv = row["Category"].strip()
            item_name = row["Item"].strip()

            # Skip empty rows
            if not category_csv or not item_name:
                continue

            # Deduplicate by (category, item_name)
            dedup_key = (category_csv, item_name)
            if dedup_key in seen_items:
                continue
            seen_items.add(dedup_key)

            # Map category
            category_name = CATEGORY_MAP.get(category_csv)
            if category_name is None:
                print(f"Warning: Unknown category '{category_csv}', skipping item '{item_name}'")
                continue

            # Extract size
            size = extract_size(item_name)

            # Generate item_id
            item_id = slugify(item_name)

            items.append({
                "item_id": item_id,
                "name": item_name,
                "category_name": category_name.value,
                "size": size.value,
                "quantity": 1,
                "modifiers": [],
            })

    menu_data = {
        "menu_id": "mcd-main-menu",
        "name": "McDonald's Menu",
        "items": items,
    }

    # Validate against Pydantic model
    menu = Menu.model_validate(menu_data)
    print(f"Validation passed: {len(menu.items)} items")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(menu_data, f, indent=2)

    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
```

## Success Criteria

### Automated Verification
```bash
# Script runs without errors
uv run python scripts/csv_to_json.py

# JSON is valid
uv run python -c "import json; json.load(open('menus/transformed-data/menu.json'))"

# Validates against Menu model
uv run python -c "from src.models import Menu; import json; Menu.model_validate(json.load(open('menus/transformed-data/menu.json')))"
```

### Manual Verification
- [ ] All 9 categories represented in items
- [ ] No duplicate items in output
- [ ] Size correctly extracted for sized items
- [ ] Piece count items have correct size mapping
- [ ] Item IDs are valid kebab-case slugs
- [ ] JSON is human-readable with proper formatting

## Testing Strategy

### Unit Tests (optional follow-up)
- `slugify()` with various inputs
- `extract_size()` with all size patterns
- Category mapping completeness

### Integration Test
- Run script end-to-end
- Validate output loads into Pydantic model

## References

- Source CSV: `menus/transformed-data/mcdonalds-menu-items-revised-full.csv`
- Pydantic Models: `src/models.py`
- Enums: `src/enums.py`
- Original Plan: `plans/2026-02-01-csv-to-json-transformation.md`
