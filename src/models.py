from pydantic import BaseModel, Field
from enums import Size, CategoryName
import random


class Modifier(BaseModel):
    modifier_id: str
    name: str
    allowed_categories: list[CategoryName] = Field(default_factory=list)


class Item(BaseModel):
    item_id: str
    name: str
    category_name: CategoryName
    size: Size = Field(default=Size.MEDIUM)
    quantity: int = Field(default=1, ge=1)
    modifiers: list[Modifier] = Field(default_factory=list)


class Order(BaseModel):
    order_id: int = Field(default_factory=lambda: random.randint(1, 1000))
    items: list[Item] = Field(default_factory=list)


# class Combo(BaseModel):
#     combo_id: str
#     items: list[Item]
#     quantity: int = Field(default=1, ge=1)
