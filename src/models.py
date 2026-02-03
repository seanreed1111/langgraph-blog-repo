import json
from pathlib import Path
from typing import Self
import uuid

from pydantic import BaseModel, Field, model_validator

from enums import Size, CategoryName


class Modifier(BaseModel):
    modifier_id: str
    name: str


class Location(BaseModel):
    id: str
    name: str
    address: str
    city: str
    state: str
    zip: str
    country: str


class Item(BaseModel):
    item_id: str
    name: str
    category_name: CategoryName
    default_size: Size = Field(default=Size.REGULAR)
    size: Size | None = Field(default=None)
    quantity: int = Field(default=1, ge=1)
    modifiers: list[Modifier] = Field(
        default_factory=list
    )  # Customer selections (for orders)
    available_modifiers: list[Modifier] = Field(default_factory=list)  # Menu options

    @model_validator(mode="after")
    def set_size_from_default(self) -> Self:
        if self.size is None:
            self.size = self.default_size
        return self


class Order(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    items: list[Item] = Field(default_factory=list)


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
