import random
import logging

# Define categories for different clothing types
UPPER_BODY = {"Polo", "T-shirt", "Sport T-shirt", "Shirt"}
LOWER_BODY_HOT = {"Cargo Pants", "Chinos", "Jeans", "Joggers", "Pants", "Shorts"}
LOWER_BODY_COLD = {"Cargo Pants", "Chinos", "Jeans", "Joggers", "Pants"}
OUTERWEAR = {"Crewneck", "Hoodie", "Fleece", "Jacket", "Overcoat", "Overshirt", "Suit"}
FOOTWEAR = {"Shoes", "Sneakers"}

# Define color families for compatibility checks
COLOR_FAMILIES = {
    "black": {"black"},
    "white": {"white"},
    "blue": {"blue", "lightblue"},
    "gray": {"gray", "lightgray", "silver"},
    "brown": {"brown", "beige", "khaki"},
    "green": {"green"},
    "red": {"red"},
    "yellow": {"yellow"},
    "purple": {"purple", "pink"},
    "multicolor": {"multi-color", "multicolor"},
}

# Neutrals are compatible with most other colors
NEUTRAL_FAMILIES = {"black", "white", "gray", "brown"}


def get_color_family(color):
    """Finds the family a specific color belongs to."""
    if not color:
        return None
    c = color.lower().replace(" ", "")
    for family, colors in COLOR_FAMILIES.items():
        if c in colors:
            return family
    return None


def get_families_from_colors(color_list):
    """Gets all unique color families from a list of colors."""
    return {fam for c in color_list if (fam := get_color_family(c))}


def colors_compatible(colors1, colors2):
    """Checks if two lists of colors are compatible."""
    fams1 = get_families_from_colors(colors1)
    fams2 = get_families_from_colors(colors2)
    if not fams1 or not fams2:
        return False
    # Compatible if they share a family or if one is neutral
    if fams1.intersection(fams2):
        return True
    if fams1.intersection(NEUTRAL_FAMILIES) or fams2.intersection(NEUTRAL_FAMILIES):
        return True
    return False


def filter_items_by_aesthetic(items, desired_aesthetic):
    """Filters a list of items to only include those matching the desired aesthetic."""
    if not desired_aesthetic:
        return items
    return [
        item for item in items
        if any(a.lower() == desired_aesthetic.lower() for a in item.get("aesthetic", []))
    ]


def filter_items_by_color(items, base_colors):
    """Filters items to find ones with colors compatible with the base colors."""
    if not base_colors:
        return items
    return [
        item for item in items
        if colors_compatible(base_colors, item.get("color", []))
    ]


def pick_one(items):
    """Randomly picks one item from a list."""
    return random.choice(items) if items else None


def pick_upper_body(items, desired_aesthetic, base_colors):
    candidates = [i for i in items if i["category"] in UPPER_BODY]
    candidates = filter_items_by_aesthetic(candidates, desired_aesthetic)
    candidates = filter_items_by_color(candidates, base_colors)
    return pick_one(candidates)


def pick_lower_body(items, hot, desired_aesthetic, base_colors):
    lower_body_cats = LOWER_BODY_HOT if hot else LOWER_BODY_COLD
    candidates = [i for i in items if i["category"] in lower_body_cats]
    candidates = filter_items_by_aesthetic(candidates, desired_aesthetic)
    candidates = filter_items_by_color(candidates, base_colors)
    return pick_one(candidates)


def pick_outerwear(items, desired_aesthetic, base_colors):
    candidates = [i for i in items if i["category"] in OUTERWEAR]
    candidates = filter_items_by_aesthetic(candidates, desired_aesthetic)
    candidates = filter_items_by_color(candidates, base_colors)
    return pick_one(candidates)


def pick_footwear(items, desired_aesthetic, base_colors):
    candidates = [i for i in items if i["category"] in FOOTWEAR]
    candidates = filter_items_by_aesthetic(candidates, desired_aesthetic)
    candidates = filter_items_by_color(candidates, base_colors)
    return pick_one(candidates)


def build_outfit(items, hot, desired_aesthetic, washed_required="Done"):
    """
    Builds a complete, color-coordinated outfit from available items.
    """
    # 1. Filter for clean clothes first, handling empty/None values for "washed"
    filtered_items = [
        item for item in items
        if (item.get("Washed") or "").lower() == washed_required.lower()
    ]
    if not filtered_items:
        logging.warning(f"No items found with washed status '{washed_required}'.")
        return []

    # 2. Pick an upper body item to set the base color scheme
    upper = pick_upper_body(filtered_items, desired_aesthetic, base_colors=[])
    if not upper:
        logging.warning("No upper body garment found for outfit.")
        return []

    base_colors = upper.get("color", [])

    # 3. Pick other items that are compatible with the upper body item
    lower = pick_lower_body(filtered_items, hot, desired_aesthetic, base_colors)
    if not lower:
        logging.warning("No lower body garment found for outfit.")
        return []

    footwear = pick_footwear(filtered_items, desired_aesthetic, base_colors)
    if not footwear:
        logging.warning("No footwear found for outfit.")
        return []

    outfit = [upper]

    # 4. Add outerwear if it's not hot
    if not hot:
        outer = pick_outerwear(filtered_items, desired_aesthetic, base_colors)
        if outer:
            outfit.append(outer)

    outfit.append(lower)
    outfit.append(footwear)

    # 5. Sort the outfit for logical presentation (top, outerwear, bottom, shoes)
    def sort_key(item):
        cat = item["category"]
        if cat in UPPER_BODY: return 0
        if cat in OUTERWEAR: return 1
        if cat in LOWER_BODY_HOT or cat in LOWER_BODY_COLD: return 2
        if cat in FOOTWEAR: return 3
        return 4

    return sorted(outfit, key=sort_key)