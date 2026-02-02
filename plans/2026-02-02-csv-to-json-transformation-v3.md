# CSV to JSON Transformation Plan (v3) - With Modifier Extraction

> **Status:** DRAFT
> **Supersedes:** `2026-02-02-csv-to-json-transformation-v2.md`
> **Design Decisions:** `2026-02-02-modifier-extraction-questions.md`

## Overview

Transform `menus/transformed-data/mcdonalds-menu-items-revised-full.csv` into a JSON file that:
1. Collapses variant items into a single base Item
2. Extracts available Modifiers from the "with X" pattern
3. Validates against updated Pydantic models

## Design Decisions Summary

| Decision | Choice |
|----------|--------|
| "with X" pattern | Extract X as modifier on base item |
| "with X and Y" pattern | Extract as two separate modifiers |
| Prefix variants (Bacon McDouble) | Separate items, NOT modifiers |
| Suffix variants (Crispy/Grilled Chicken) | Modifiers |
| Compound items (Double Quarter Pounder) | Separate item, "with X" still extracts modifiers |
| Coffee & Tea | Only "with X" patterns (not flavor prefixes) |
| Item/Modifier overlap | Same thing can be both |
| Modifier storage | Item level: `available_modifiers: list[Modifier]` |

## Pattern Examples

### Pattern 1: "with X" → Modifier Extraction

```
CSV Input:
  Breakfast,Sausage McMuffin
  Breakfast,Sausage McMuffin with Egg
  Breakfast,Sausage McMuffin with Egg Whites

Output:
  Item: "Sausage McMuffin"
  available_modifiers: [Egg, Egg Whites]
```

### Pattern 2: "with X and Y" → Multiple Modifiers

```
CSV Input:
  Breakfast,Big Breakfast
  Breakfast,Big Breakfast with Egg Whites
  Breakfast,Big Breakfast with Hotcakes
  Breakfast,Big Breakfast with Hotcakes and Egg Whites

Output:
  Item: "Big Breakfast"
  available_modifiers: [Egg Whites, Hotcakes]
```

### Pattern 3: Suffix Variants → Modifiers

```
CSV Input:
  Chicken & Fish,Premium McWrap Chicken & Bacon, Crispy Chicken
  Chicken & Fish,Premium McWrap Chicken & Bacon, Grilled Chicken

Output:
  Item: "Premium McWrap Chicken & Bacon"
  available_modifiers: [Crispy Chicken, Grilled Chicken]
```

### Pattern 4: Prefix Variants → Separate Items

```
CSV Input:
  Beef & Pork,McDouble
  Beef & Pork,Bacon McDouble

Output:
  Item: "McDouble" (no modifiers from this)
  Item: "Bacon McDouble" (separate item)
```

### Pattern 5: Compound Items

```
CSV Input:
  Beef & Pork,Quarter Pounder
  Beef & Pork,Quarter Pounder with Cheese
  Beef & Pork,Double Quarter Pounder with Cheese

Output:
  Item: "Quarter Pounder"
  available_modifiers: [Cheese, Bacon & Cheese, Bacon Habanero Ranch, Deluxe]

  Item: "Double Quarter Pounder" (separate base item)
  available_modifiers: [Cheese]
```

## Required Model Changes

### Current `Item` model (src/models.py)

```python
class Item(BaseModel):
    item_id: str
    name: str
    category_name: CategoryName
    size: Size = Field(default=Size.MEDIUM)
    quantity: int = Field(default=1, ge=1)
    modifiers: list[Modifier] = Field(default_factory=list)  # Selected modifiers
```

### Add new field

```python
class Item(BaseModel):
    item_id: str
    name: str
    category_name: CategoryName
    size: Size = Field(default=Size.MEDIUM)
    quantity: int = Field(default=1, ge=1)
    modifiers: list[Modifier] = Field(default_factory=list)  # Selected modifiers
    available_modifiers: list[Modifier] = Field(default_factory=list)  # What CAN be added
```

**Semantic distinction:**
- `modifiers` = what the customer has chosen (for an order)
- `available_modifiers` = what options exist (for the menu)

## Transformation Algorithm

### Step 1: Parse and Group

```
1. Read all CSV rows
2. For each row, detect if it's a variant:
   - Check for "with X" or "with X and Y" pattern
   - Check for suffix ", Crispy Chicken" or ", Grilled Chicken"
3. Group variants by base item name + category
```

### Step 2: Extract Modifiers

```
For each group:
  1. Identify base item (shortest name without "with" or variant suffix)
  2. For each variant:
     - Extract modifier name(s) from pattern
     - Add to base item's available_modifiers
  3. Deduplicate modifiers
```

### Step 3: Handle Size Variants

```
Size variants (", Small", ", Medium", ", Large") create separate items:
  - "Coca-Cola Classic, Small" → Item with size=small
  - "Coca-Cola Classic, Large" → Item with size=large

These are NOT collapsed - each size is a distinct menu item.
```

## Detailed Parsing Rules

### Rule 1: "with X" Detection

```python
pattern = r"^(.+?)\s+with\s+(.+?)(?:,\s*(Small|Medium|Large|Regular|Kids))?$"

# Examples:
# "Sausage McMuffin with Egg" → base="Sausage McMuffin", modifier="Egg"
# "Big Breakfast with Hotcakes, Regular" → base="Big Breakfast", modifier="Hotcakes", size="Regular"
```

### Rule 2: "with X and Y" Detection

```python
pattern = r"^(.+?)\s+with\s+(.+?)\s+and\s+(.+?)(?:,\s*(Small|Medium|Large|Regular|Kids))?$"

# Example:
# "Big Breakfast with Hotcakes and Egg Whites" → base="Big Breakfast", modifiers=["Hotcakes", "Egg Whites"]
```

### Rule 3: Suffix Variant Detection (Crispy/Grilled)

```python
pattern = r"^(.+?),\s*(Crispy Chicken|Grilled Chicken)$"

# Example:
# "Premium McWrap Chicken & Bacon, Crispy Chicken" → base="Premium McWrap Chicken & Bacon", modifier="Crispy Chicken"
```

### Rule 4: Size Suffix (NOT a modifier)

```python
size_pattern = r",\s*(Small|Medium|Large|Kids|Regular|Snack|\d+\s*pc)$"

# These indicate size, not modifiers
# "Coca-Cola Classic, Small" → size extraction only
```

## Expected Output Structure

```json
{
  "menu_id": "mcd-main-menu",
  "name": "McDonald's Menu",
  "items": [
    {
      "item_id": "sausage-mcmuffin",
      "name": "Sausage McMuffin",
      "category_name": "breakfast",
      "size": "medium",
      "quantity": 1,
      "modifiers": [],
      "available_modifiers": [
        {"modifier_id": "egg", "name": "Egg"},
        {"modifier_id": "egg-whites", "name": "Egg Whites"}
      ]
    },
    {
      "item_id": "quarter-pounder",
      "name": "Quarter Pounder",
      "category_name": "beef-pork",
      "size": "medium",
      "quantity": 1,
      "modifiers": [],
      "available_modifiers": [
        {"modifier_id": "cheese", "name": "Cheese"},
        {"modifier_id": "bacon-cheese", "name": "Bacon & Cheese"},
        {"modifier_id": "bacon-habanero-ranch", "name": "Bacon Habanero Ranch"},
        {"modifier_id": "deluxe", "name": "Deluxe"}
      ]
    },
    {
      "item_id": "double-quarter-pounder",
      "name": "Double Quarter Pounder",
      "category_name": "beef-pork",
      "size": "medium",
      "quantity": 1,
      "modifiers": [],
      "available_modifiers": [
        {"modifier_id": "cheese", "name": "Cheese"}
      ]
    },
    {
      "item_id": "coca-cola-classic-small",
      "name": "Coca-Cola Classic",
      "category_name": "beverages",
      "size": "small",
      "quantity": 1,
      "modifiers": [],
      "available_modifiers": []
    }
  ]
}
```

## Implementation Phases

### Phase 1: Update Pydantic Model

Add `available_modifiers` field to `Item` in `src/models.py`.

### Phase 2: Create Transformation Script

File: `scripts/csv_to_json_v3.py`

```python
"""Transform McDonald's menu CSV to JSON with modifier extraction."""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import Menu, Item, Modifier
from enums import CategoryName, Size


def slugify(text: str) -> str:
    """Convert text to kebab-case slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


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

SIZE_PATTERN = re.compile(r",\s*(Small|Medium|Large|Kids|Regular|Snack|\d+\s*pc)$", re.IGNORECASE)
WITH_AND_PATTERN = re.compile(r"^(.+?)\s+with\s+(.+?)\s+and\s+(.+)$", re.IGNORECASE)
WITH_PATTERN = re.compile(r"^(.+?)\s+with\s+(.+)$", re.IGNORECASE)
SUFFIX_VARIANT_PATTERN = re.compile(r"^(.+?),\s*(Crispy Chicken|Grilled Chicken)$", re.IGNORECASE)


def extract_size(item_name: str) -> tuple[str, Size]:
    """Extract size from item name, return (name_without_size, Size)."""
    match = SIZE_PATTERN.search(item_name)
    if not match:
        return item_name, Size.MEDIUM

    size_str = match.group(1).lower()
    name_without_size = item_name[:match.start()].strip()

    size_map = {
        "small": Size.SMALL,
        "medium": Size.MEDIUM,
        "large": Size.LARGE,
        "kids": Size.SMALL,
        "regular": Size.MEDIUM,
        "snack": Size.SNACK,
    }

    # Handle piece counts
    if "pc" in size_str:
        pc = int(re.search(r"(\d+)", size_str).group(1))
        pc_map = {4: Size.SNACK, 6: Size.SMALL, 10: Size.MEDIUM, 20: Size.LARGE, 40: Size.LARGE}
        return name_without_size, pc_map.get(pc, Size.MEDIUM)

    return name_without_size, size_map.get(size_str, Size.MEDIUM)


def parse_item(item_name: str) -> dict:
    """Parse item name into base name, modifiers, and size."""
    # First extract size
    name_no_size, size = extract_size(item_name)

    # Check for suffix variant (Crispy/Grilled Chicken)
    suffix_match = SUFFIX_VARIANT_PATTERN.match(name_no_size)
    if suffix_match:
        return {
            "base_name": suffix_match.group(1).strip(),
            "modifiers": [suffix_match.group(2).strip()],
            "size": size,
            "original_name": item_name,
        }

    # Check for "with X and Y" pattern
    with_and_match = WITH_AND_PATTERN.match(name_no_size)
    if with_and_match:
        return {
            "base_name": with_and_match.group(1).strip(),
            "modifiers": [with_and_match.group(2).strip(), with_and_match.group(3).strip()],
            "size": size,
            "original_name": item_name,
        }

    # Check for "with X" pattern
    with_match = WITH_PATTERN.match(name_no_size)
    if with_match:
        return {
            "base_name": with_match.group(1).strip(),
            "modifiers": [with_match.group(2).strip()],
            "size": size,
            "original_name": item_name,
        }

    # No pattern matched - this is a base item
    return {
        "base_name": name_no_size,
        "modifiers": [],
        "size": size,
        "original_name": item_name,
    }


def main():
    csv_path = Path("menus/transformed-data/mcdonalds-menu-items-revised-full.csv")
    output_path = Path("menus/transformed-data/menu.json")

    # Group items by (category, base_name, size)
    # Key: (category, base_name, size) -> list of modifier names
    item_groups = defaultdict(lambda: {"modifiers": set(), "seen": False})

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category_csv = row["Category"].strip()
            item_name = row["Item"].strip()

            if not category_csv or not item_name:
                continue

            category = CATEGORY_MAP.get(category_csv)
            if not category:
                print(f"Warning: Unknown category '{category_csv}'")
                continue

            parsed = parse_item(item_name)
            key = (category, parsed["base_name"], parsed["size"])

            if parsed["modifiers"]:
                item_groups[key]["modifiers"].update(parsed["modifiers"])
            else:
                item_groups[key]["seen"] = True

    # Build items list
    items = []
    for (category, base_name, size), data in item_groups.items():
        modifiers = [
            {"modifier_id": slugify(m), "name": m}
            for m in sorted(data["modifiers"])
        ]

        # Generate item_id (include size if not medium)
        item_id = slugify(base_name)
        if size != Size.MEDIUM:
            item_id = f"{item_id}-{size.value}"

        items.append({
            "item_id": item_id,
            "name": base_name,
            "category_name": category.value,
            "size": size.value,
            "quantity": 1,
            "modifiers": [],
            "available_modifiers": modifiers,
        })

    # Sort by category then name
    items.sort(key=lambda x: (x["category_name"], x["name"]))

    menu_data = {
        "menu_id": "mcd-main-menu",
        "name": "McDonald's Menu",
        "items": items,
    }

    # Validate
    menu = Menu.model_validate(menu_data)
    print(f"Validation passed: {len(menu.items)} items")

    # Count items with modifiers
    with_mods = sum(1 for i in items if i["available_modifiers"])
    print(f"Items with available_modifiers: {with_mods}")

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
uv run python scripts/csv_to_json_v3.py

# Validates against Menu model
uv run python -c "from src.models import Menu; import json; Menu.model_validate(json.load(open('menus/transformed-data/menu.json')))"
```

### Manual Verification

- [ ] "Sausage McMuffin" has modifiers: [Egg, Egg Whites]
- [ ] "Quarter Pounder" has modifiers: [Cheese, Bacon & Cheese, Bacon Habanero Ranch, Deluxe]
- [ ] "Double Quarter Pounder" is a separate item with modifier: [Cheese]
- [ ] "McDouble" and "Bacon McDouble" are separate items (no modifier relationship)
- [ ] "Premium McWrap Chicken & Bacon" has modifiers: [Crispy Chicken, Grilled Chicken]
- [ ] Coffee items only extract "with X" patterns (not flavor prefixes)
- [ ] Size variants remain separate items
- [ ] Fewer total items than v2 (due to collapsing)

## Item Count Estimate

| Category | v2 Items (approx) | v3 Items (approx) | Reduction |
|----------|-------------------|-------------------|-----------|
| Breakfast | ~36 | ~20 | ~16 |
| Beef & Pork | ~17 | ~12 | ~5 |
| Chicken & Fish | ~27 | ~15 | ~12 |
| Coffee & Tea | ~100 | ~90 | ~10 |
| Other | ~89 | ~89 | 0 |
| **Total** | ~269 | ~226 | ~43 |

## Edge Cases

### Handled

1. **Size + Modifier**: "Big Breakfast with Hotcakes, Regular" → extracts both
2. **Multiple modifiers**: "with X and Y" → two separate modifiers
3. **Standalone items that are also modifiers**: "Hotcakes" exists as Item AND as modifier on "Big Breakfast"

### Not Handled (by design)

1. **Prefix flavors**: "Caramel Latte" stays separate from "Latte"
2. **Prefix variants**: "Bacon McDouble" stays separate from "McDouble"
3. **"Nonfat" prefix**: "Nonfat Latte" stays separate from "Latte"

## References

- Source CSV: `menus/transformed-data/mcdonalds-menu-items-revised-full.csv`
- Pydantic Models: `src/models.py`
- Design Decisions: `plans/2026-02-02-modifier-extraction-questions.md`
- Previous Plan: `plans/2026-02-02-csv-to-json-transformation-v2.md`
