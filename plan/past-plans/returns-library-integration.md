# Returns Library Integration with Pydantic Models

Suggestions for integrating the `returns` library with the McDonald's data models.

## 1. Result for Validation/Lookup Operations

```python
from returns.result import Result, Success, Failure

def find_item_by_id(menu: Menu, item_id: str) -> Result[Item, str]:
    """Returns Success(Item) or Failure with error message."""
    for category in menu.allowed_categories:
        # lookup logic...
        pass
    return Failure(f"Item {item_id} not found")

def validate_modifier_for_category(modifier: Modifier, category: Category) -> Result[Modifier, str]:
    if category in modifier.restrictions:
        return Failure(f"Modifier '{modifier.name}' not allowed for {category.name}")
    return Success(modifier)
```

## 2. Maybe for Optional Lookups

```python
from returns.maybe import Maybe, Some, Nothing

def get_default_size(item_name: str) -> Maybe[Size]:
    """Some items might have special default sizes."""
    defaults = {"McFlurry": Size.SNACK, "Large Fries": Size.LARGE}
    return Maybe.from_optional(defaults.get(item_name))
```

## 3. Pipeline Operations with `.map()` and `.bind()`

```python
from returns.result import Result, Success
from returns.pipeline import flow

def process_order_item(raw_data: dict) -> Result[Item, str]:
    return flow(
        raw_data,
        validate_required_fields,   # -> Result[dict, str]
        lambda d: d.bind(parse_item),  # -> Result[Item, str]
        lambda r: r.bind(validate_modifiers),
    )
```

## 4. ResultE for Exception-Based Validation

```python
from returns.result import ResultE, safe

@safe
def parse_item_from_json(data: dict) -> Item:
    """Automatically catches Pydantic ValidationError."""
    return Item(**data)

# Returns ResultE[Item] (Success or Failure with exception)
```

## 5. Combining with Pydantic Validators

```python
from returns.result import Result, Success, Failure
from pydantic import model_validator

class Item(BaseModel):
    # ... fields ...

    @model_validator(mode="after")
    def validate_combo_compatibility(self) -> "Item":
        # Use returns internally, unwrap for Pydantic
        result = check_modifier_compatibility(self)
        match result:
            case Failure(err):
                raise ValueError(err)
            case Success(item):
                return item
```

## 6. IO for Menu Loading Side Effects

```python
from returns.io import IO, impure_safe

@impure_safe
def load_menu_from_file(path: str) -> Menu:
    """Wraps file I/O, returns IOResultE[Menu]."""
    with open(path) as f:
        return Menu.model_validate_json(f.read())
```

## Key Integration Points

| Use Case | Returns Type | Example |
|----------|-------------|---------|
| Item/Category lookup | `Result[T, str]` | Finding items by ID |
| Optional data | `Maybe[T]` | Default sizes, optional modifiers |
| Parsing external input | `ResultE[T]` / `@safe` | JSON to Pydantic |
| File/DB operations | `IOResultE[T]` | Loading menus |
| Chained validations | `.bind()` / `flow()` | Order processing pipeline |

## Benefits

The main benefit is making failure modes explicit in your type signatures rather than relying on exceptions or `Optional` with `None` checks.
