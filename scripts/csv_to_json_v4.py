"""
Transform McDonald's menu CSV to JSON with modifier extraction.

Usage:
    uv run python scripts/csv_to_json_v4.py

Output:
    menus/transformed-data/menu.json
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enums import CategoryName, Size
from models import Item, Menu, Modifier


# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_PATH = Path("menus/transformed-data/mcdonalds-menu-items-revised-full.csv")
OUTPUT_PATH = Path("menus/transformed-data/menu.json")

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

SIZE_WORDS_MAP = {
    "small": Size.SMALL,
    "medium": Size.MEDIUM,
    "large": Size.LARGE,
    "kids": Size.SMALL,
    "regular": Size.MEDIUM,
    "snack": Size.SNACK,
}

PC_SIZE_MAP = {
    4: Size.SNACK,
    6: Size.SMALL,
    10: Size.MEDIUM,
    20: Size.LARGE,
    40: Size.LARGE,
}

# Items that use "with" but should NOT have modifiers extracted
NON_COLLAPSIBLE_PREFIXES = [
    "McFlurry with",
]


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Size patterns
SUFFIX_SIZE_PATTERN = re.compile(
    r"^(.+?),\s*(Small|Medium|Large|Kids|Regular|Snack|\d+\s*pc)$", re.IGNORECASE
)
PREFIX_SIZE_PATTERN = re.compile(r"^(Small|Medium|Large|Kids)\s+(.+)$", re.IGNORECASE)

# Modifier patterns (order matters - check most specific first)
WITH_AND_PATTERN = re.compile(r"^(.+?)\s+with\s+(.+?)\s+and\s+(.+)$", re.IGNORECASE)
WITH_PATTERN = re.compile(r"^(.+?)\s+with\s+(.+)$", re.IGNORECASE)
SUFFIX_VARIANT_PATTERN = re.compile(
    r"^(.+?),\s*(Crispy Chicken|Grilled Chicken)$", re.IGNORECASE
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def slugify(text: str) -> str:
    """Convert text to kebab-case slug for IDs."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def is_non_collapsible(item_name: str) -> bool:
    """Check if item should NOT have modifiers extracted."""
    for prefix in NON_COLLAPSIBLE_PREFIXES:
        if item_name.startswith(prefix):
            return True
    return False


def extract_size(item_name: str) -> tuple[str, Size]:
    """
    Extract size from item name.

    Returns: (name_without_size, Size)
    """
    # Check suffix pattern first (e.g., "Coca-Cola, Small")
    suffix_match = SUFFIX_SIZE_PATTERN.match(item_name)
    if suffix_match:
        name = suffix_match.group(1).strip()
        size_str = suffix_match.group(2).lower()

        # Handle piece counts
        if "pc" in size_str:
            pc_match = re.search(r"(\d+)", size_str)
            if pc_match:
                pc = int(pc_match.group(1))
                return name, PC_SIZE_MAP.get(pc, Size.MEDIUM)

        return name, SIZE_WORDS_MAP.get(size_str, Size.MEDIUM)

    # Check prefix pattern (e.g., "Small French Fries")
    prefix_match = PREFIX_SIZE_PATTERN.match(item_name)
    if prefix_match:
        size_str = prefix_match.group(1).lower()
        name = prefix_match.group(2).strip()
        return name, SIZE_WORDS_MAP.get(size_str, Size.MEDIUM)

    # No size pattern found
    return item_name, Size.MEDIUM


def parse_item(item_name: str) -> dict:
    """
    Parse item name into components.

    Returns dict with:
        - base_name: str
        - modifiers: list[str]
        - size: Size
        - is_base_item: bool (True if no modifiers extracted)
    """
    # First extract size
    name_no_size, size = extract_size(item_name)

    # Check if this item should not have modifiers extracted
    if is_non_collapsible(name_no_size):
        return {
            "base_name": name_no_size,
            "modifiers": [],
            "size": size,
            "is_base_item": True,
        }

    # Check for suffix variant (Crispy/Grilled Chicken) - BEFORE checking "with" patterns
    # This handles "Premium McWrap..., Crispy Chicken"
    suffix_match = SUFFIX_VARIANT_PATTERN.match(name_no_size)
    if suffix_match:
        return {
            "base_name": suffix_match.group(1).strip(),
            "modifiers": [suffix_match.group(2).strip()],
            "size": size,
            "is_base_item": False,
        }

    # Check for "with X and Y" pattern
    with_and_match = WITH_AND_PATTERN.match(name_no_size)
    if with_and_match:
        return {
            "base_name": with_and_match.group(1).strip(),
            "modifiers": [
                with_and_match.group(2).strip(),
                with_and_match.group(3).strip(),
            ],
            "size": size,
            "is_base_item": False,
        }

    # Check for "with X" pattern
    with_match = WITH_PATTERN.match(name_no_size)
    if with_match:
        return {
            "base_name": with_match.group(1).strip(),
            "modifiers": [with_match.group(2).strip()],
            "size": size,
            "is_base_item": False,
        }

    # No pattern matched - this is a base item
    return {
        "base_name": name_no_size,
        "modifiers": [],
        "size": size,
        "is_base_item": True,
    }


# =============================================================================
# MAIN TRANSFORMATION
# =============================================================================


def main():
    print(f"Reading CSV: {CSV_PATH}")

    # Track seen items to handle duplicates
    seen_items: set[tuple[str, str]] = set()

    # Group items by (category, base_name, size)
    # Value: {"modifiers": set(), "has_base": bool}
    item_groups: dict[tuple, dict] = defaultdict(
        lambda: {"modifiers": set(), "has_base": False}
    )

    # Track statistics
    stats = {
        "total_rows": 0,
        "skipped_empty": 0,
        "skipped_duplicate": 0,
        "skipped_unknown_category": 0,
        "processed": 0,
    }

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            stats["total_rows"] += 1

            # Clean row data
            category_csv = row.get("Category", "").strip()
            item_name = row.get("Item", "").strip()

            # Handle malformed CSV where Item field contains unquoted commas
            # These get parsed as extra columns with None keys
            if None in row and row[None]:
                # Concatenate all extra fields back into item_name
                extra_fields = row[None] if isinstance(row[None], list) else [row[None]]
                item_name = item_name + "," + ",".join(extra_fields)

            # Skip empty rows
            if not category_csv or not item_name:
                stats["skipped_empty"] += 1
                continue

            # Skip duplicates
            dup_key = (category_csv, item_name)
            if dup_key in seen_items:
                stats["skipped_duplicate"] += 1
                continue
            seen_items.add(dup_key)

            # Map category
            category = CATEGORY_MAP.get(category_csv)
            if not category:
                print(f"  Warning: Unknown category '{category_csv}' - skipping")
                stats["skipped_unknown_category"] += 1
                continue

            # Parse item
            parsed = parse_item(item_name)
            group_key = (category, parsed["base_name"], parsed["size"])

            # Update group
            if parsed["modifiers"]:
                item_groups[group_key]["modifiers"].update(parsed["modifiers"])
            if parsed["is_base_item"]:
                item_groups[group_key]["has_base"] = True

            stats["processed"] += 1

    print("\nCSV Processing Stats:")
    print(f"  Total rows: {stats['total_rows']}")
    print(f"  Skipped empty: {stats['skipped_empty']}")
    print(f"  Skipped duplicate: {stats['skipped_duplicate']}")
    print(f"  Skipped unknown category: {stats['skipped_unknown_category']}")
    print(f"  Processed: {stats['processed']}")

    # Build items list
    items = []
    for (category, base_name, size), data in item_groups.items():
        # Create available_modifiers list
        available_modifiers = [
            Modifier(modifier_id=slugify(m), name=m) for m in sorted(data["modifiers"])
        ]

        # Generate item_id (include size if not medium)
        item_id = slugify(base_name)
        if size != Size.MEDIUM:
            item_id = f"{item_id}-{size.value}"

        item = Item(
            item_id=item_id,
            name=base_name,
            category_name=category,
            size=size,
            quantity=1,
            modifiers=[],
            available_modifiers=available_modifiers,
        )
        items.append(item)

    # Sort by category then name
    items.sort(key=lambda x: (x.category_name.value, x.name))

    print(f"\nGenerated {len(items)} items")

    # Count items with modifiers
    with_mods = sum(1 for i in items if i.available_modifiers)
    print(f"Items with available_modifiers: {with_mods}")

    # Build menu
    menu = Menu(
        menu_id="mcd-main-menu",
        name="McDonald's Menu",
        items=items,
    )

    # Validate by converting to dict and back
    print("\nValidating with Pydantic...")
    menu_dict = menu.model_dump()
    Menu.model_validate(menu_dict)
    print("Validation passed!")

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(menu_dict, f, indent=2)

    print(f"\nOutput written to: {OUTPUT_PATH}")

    # Print sample output for verification
    print("\n--- Sample Items ---")
    sample_items = [
        "Sausage McMuffin",
        "Quarter Pounder",
        "Double Quarter Pounder",
        "McDouble",
        "Bacon McDouble",
        "Premium McWrap Chicken & Bacon",
        "McFlurry with M&Ms Candies",
    ]
    for sample_name in sample_items:
        for item in items:
            if item.name == sample_name:
                mods = [m.name for m in item.available_modifiers]
                print(
                    f"  {item.name} ({item.size.value}): {mods if mods else '(no modifiers)'}"
                )
                break


if __name__ == "__main__":
    main()
