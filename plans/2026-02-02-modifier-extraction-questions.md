# Modifier Extraction - Clarifying Questions

> **Status:** AWAITING REVIEW
> **Related Plan:** `2026-02-02-csv-to-json-transformation-v2.md`

## Context

Updating the CSV-to-JSON transformation to extract Modifier objects from item name patterns. Before creating the updated plan, the following design decisions need clarification.

---

## Question 1: Pattern Detection Scope

The "with X" pattern is clear:
- "Quarter Pounder with Cheese" → base: Quarter Pounder, modifier: Cheese
- "Sausage McMuffin with Egg" → base: Sausage McMuffin, modifier: Egg

**What about these other patterns in the CSV?**

| Pattern | Examples | Treat as modifiers? |
|---------|----------|---------------------|
| Prefix variants | "Bacon McDouble" vs "McDouble", "Nonfat Latte" vs "Latte" | 
| Suffix variants | "Premium Crispy Chicken" vs "Premium Grilled Chicken" | 
| Compound | "Double Quarter Pounder with Cheese" | Separate item or modifier on QP? 

**Your answer:**

| Pattern | Examples | Treat as modifiers? |
|---------|----------|---------------------|
| Prefix variants | "Bacon McDouble" vs "McDouble", "Nonfat Latte" vs "Latte" | **Separate Items** 
| Suffix variants | "Premium Crispy Chicken" vs "Premium Grilled Chicken" | Crispy Chicken and Grilled Chicken are both modifiers in this case ||
| Compound | "Double Quarter Pounder with Cheese" | Separate item or modifier on QP? | Double Quarter pounder is a separate item from Quarter Pounder. With Cheese as a Modifier
---

## Question 2: Dual Modifiers

"Big Breakfast with Hotcakes and Egg Whites" has two modifiers.

**How should this be represented?**

- [X] A: Separate modifiers → `available_modifiers: [Hotcakes, Egg Whites]`
- [ ] B: Combined modifier → `available_modifiers: [Hotcakes and Egg Whites]`
- [ ] C: Other (explain below)

**Your answer:**
A
---

## Question 3: Item vs Modifier Overlap

"Hotcakes" appears both as:
- A standalone item: `Breakfast,Hotcakes`
- A modifier: `Big Breakfast with Hotcakes`

**Should "Hotcakes" exist as both an Item AND a Modifier?**

- [X] A: Yes, same thing can be both
- [ ] B: No, only as Item (remove from modifiers)
- [ ] C: No, only as Modifier (remove standalone item)
- [ ] D: Other (explain below)

**Your answer:**

---

## Question 4: Coffee & Tea Complexity

The Coffee & Tea section has many layered variants:
- "Latte" → "Caramel Latte" → "Nonfat Caramel Latte"
- "Mocha" → "Mocha with Nonfat Milk"
- "Iced Coffee" → "Caramel Iced Coffee" → "Iced Coffee with Sugar Free French Vanilla Syrup"

**How should we handle this category?**

- [X] A: Focus only on clear "with X" patterns (simpler)
- [ ] B: Extract all flavor/milk variants as modifiers (complex)
- [ ] C: Keep Coffee & Tea items flat (no modifier extraction for this category)
- [ ] D: Other (explain below)

**Your answer:**

---

## Question 5: Available Modifiers Storage

The current `Modifier` model has `allowed_categories`.

**Where should available modifiers be stored?**

- [] A: Item level only → `item.available_modifiers: list[Modifier]`
- [ ] B: Menu level only → `menu.modifiers: list[Modifier]`
- [ ] C: Both (Menu has master list, Item references which apply)
- [] D: Keep current structure (Modifier has `allowed_categories`)

**Your answer:**
Modifiers should be stored only in Items. 
---

## Additional Notes

(Add any other considerations or constraints here)
