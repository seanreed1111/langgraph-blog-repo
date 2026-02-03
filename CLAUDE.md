# McDonald's Breakfast Menu Data Model

Pydantic v2 data models for a McDonald's breakfast menu drive-thru voice ordering system.

## Tech Stack

- Python 3.12+
- Pydantic v2 for data validation
- uv for package management

## Project Structure

```
src/models.py      # Pydantic models (Item, Modifier, Order, Menu)
src/enums.py       # Enums (Size, CategoryName)
menus/             # Menu data (raw CSV, transformed JSON)
thoughts/          # Design notes and requirements
```

## Commands

```bash
uv sync            # Install dependencies
uv run python      # Run Python with project dependencies
uv add <package>   # Add new dependencies (always use this instead of editing pyproject.toml)
date -Iseconds     # Get current date (use this to verify the actual date)
```

## Important: Package Management

- **Always use `uv add <package>`** to add new dependencies instead of manually editing pyproject.toml. This ensures you get the latest compatible versions.
- Run `date -Iseconds` to check the current date before suggesting Python packages or versions.
- Do not assume package versions do not exist based on training data—always verify against `uv.lock` and `pyproject.toml` in the repository, which reflect actually working versions.

## Models Overview

- `Size` - StrEnum: snack, small, medium, large
- `CategoryName` - Menu category (breakfast, beverages, coffee-tea, etc.)
- `Item` - Individual menu item with modifiers, default size is medium
- `Modifier` - Item variations (Extra Cheese, No Onions, etc.)
- `Order` - Collection of items for a customer order
- `Menu` - Full menu with items
