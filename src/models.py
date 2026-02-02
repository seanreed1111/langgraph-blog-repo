from pydantic import BaseModel, Field
from enums import Size, CategoryName
import uuid


class Modifier(BaseModel):
    modifier_id: str
    name: str


class Item(BaseModel):
    item_id: str
    name: str
    category_name: CategoryName
    size: Size = Field(default=Size.MEDIUM)
    quantity: int = Field(default=1, ge=1)
    modifiers: list[Modifier] = Field(
        default_factory=list
    )  # Customer selections (for orders)
    available_modifiers: list[Modifier] = Field(default_factory=list)  # Menu options


class Order(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    items: list[Item] = Field(default_factory=list)


class Menu(BaseModel):
    menu_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    items: list[Item]


# class Combo(BaseModel):
#     combo_id: str
#     items: list[Item]
#     quantity: int = Field(default=1, ge=1)
