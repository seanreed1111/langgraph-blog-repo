# Changelog

## [Unreleased]

### Added
- Add custom equality and hashability to `Modifier`, `Location`, `Item`, and `Menu` models (#3)
- Add quantity-based comparison operators (`<`, `<=`, `>`, `>=`) to `Item` for same-configuration items (#3)
- Add `__add__` method to `Item` to combine quantities of same-configuration items (#3)
- Add `Location` model for restaurant location data (id, name, address, city, state, zip, country) (#2)
- Add `Menu.from_json_file()` class method to load menu from JSON file path (#2)
- Add `Menu.from_dict()` class method to load menu from dictionary (#2)
- Add `REGULAR` size option to `Size` enum (#2)

### Changed
- Update `Menu` model with flattened metadata fields (menu_id, menu_name, menu_version, location) (#2)
- Reorganize menu data structure to `menus/mcdonalds/breakfast-menu/` (#2)
