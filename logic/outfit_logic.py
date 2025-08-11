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


def filter_items_by_weather(items, is_hot):
    """
    Filters items based on their weather tags and current weather conditions.
    
    Args:
        items: List of clothing items with 'weather' tag lists
        is_hot: Boolean indicating if it's hot weather
    
    Returns:
        List of items suitable for the current weather
    """
    weather_tag = "Hot" if is_hot else "Cold"
    
    suitable_items = []
    for item in items:
        item_weather_tags = item.get("weather", [])
        # Convert to lowercase for case-insensitive comparison
        item_weather_lower = [tag.lower() for tag in item_weather_tags]
        
        # Include item if:
        # 1. It has the matching weather tag, OR
        # 2. It has both Hot and Cold tags (versatile item), OR
        # 3. It has no weather tags (assume versatile)
        if (weather_tag.lower() in item_weather_lower or 
            (len(item_weather_lower) >= 2 and "hot" in item_weather_lower and "cold" in item_weather_lower) or
            len(item_weather_tags) == 0):
            suitable_items.append(item)
    
    return suitable_items


def pick_one(items):
    """Randomly picks one item from a list."""
    return random.choice(items) if items else None


def pick_upper_body(items, desired_aesthetic, base_colors):
    """Pick upper body item with aesthetic fallback - signature maintained for compatibility."""
    candidates = [i for i in items if i["category"] in UPPER_BODY]
    
    # Try with aesthetic + color filters first
    aesthetic_filtered = filter_items_by_aesthetic(candidates, desired_aesthetic)
    color_filtered = filter_items_by_color(aesthetic_filtered, base_colors)
    if color_filtered:
        return pick_one(color_filtered)
    
    # Fallback: ignore aesthetic, keep color compatibility
    color_filtered = filter_items_by_color(candidates, base_colors)
    if color_filtered:
        return pick_one(color_filtered)
    
    # Last resort: any upper body item
    return pick_one(candidates)


def pick_lower_body(items, hot, desired_aesthetic, base_colors):
    """Pick lower body item with aesthetic fallback - signature maintained, but now uses weather tags instead of category filtering."""
    # First filter by category (keep all lower body categories)
    lower_body_cats = LOWER_BODY_HOT  # Use the full set since we'll filter by weather tags
    candidates = [i for i in items if i["category"] in lower_body_cats]
    
    # Apply weather filtering based on individual item tags
    candidates = filter_items_by_weather(candidates, hot)
    
    # Try with aesthetic + color filters first
    aesthetic_filtered = filter_items_by_aesthetic(candidates, desired_aesthetic)
    color_filtered = filter_items_by_color(aesthetic_filtered, base_colors)
    if color_filtered:
        return pick_one(color_filtered)
    
    # Fallback: ignore aesthetic, keep color compatibility
    color_filtered = filter_items_by_color(candidates, base_colors)
    if color_filtered:
        return pick_one(color_filtered)
    
    # Last resort: any weather-appropriate lower body item
    return pick_one(candidates)


def pick_outerwear(items, desired_aesthetic, base_colors):
    """Pick outerwear item with aesthetic fallback - signature maintained for compatibility."""
    candidates = [i for i in items if i["category"] in OUTERWEAR]
    
    # Try with aesthetic + color filters first
    aesthetic_filtered = filter_items_by_aesthetic(candidates, desired_aesthetic)
    color_filtered = filter_items_by_color(aesthetic_filtered, base_colors)
    if color_filtered:
        return pick_one(color_filtered)
    
    # Fallback: ignore aesthetic, keep color compatibility
    color_filtered = filter_items_by_color(candidates, base_colors)
    if color_filtered:
        return pick_one(color_filtered)
    
    # Last resort: any outerwear item
    return pick_one(candidates)


def pick_footwear(items, desired_aesthetic, base_colors):
    """Pick footwear with aesthetic fallback - signature maintained for compatibility."""
    candidates = [i for i in items if i["category"] in FOOTWEAR]
    
    # Try with aesthetic + color filters first
    aesthetic_filtered = filter_items_by_aesthetic(candidates, desired_aesthetic)
    color_filtered = filter_items_by_color(aesthetic_filtered, base_colors)
    if color_filtered:
        return pick_one(color_filtered)
    
    # Fallback: ignore aesthetic, keep color compatibility
    color_filtered = filter_items_by_color(candidates, base_colors)
    if color_filtered:
        return pick_one(color_filtered)
    
    # Last resort: any footwear available
    return pick_one(candidates)


def build_outfit(items, hot, desired_aesthetic, washed_required="Done"):
    """
    Builds a complete, color-coordinated outfit from available items.
    Now uses individual weather tags while maintaining function signature.
    All categories use 3-tier fallback: aesthetic+color -> color only -> any item
    """
    # 1. Filter for clean clothes first, handling empty/None values for "washed"
    filtered_items = [
        item for item in items
        if (item.get("washed") or "").lower() == washed_required.lower()
    ]
    if not filtered_items:
        logging.warning(f"No items found with washed status '{washed_required}'.")
        return []

    # Apply weather filtering to all items upfront
    weather_filtered_items = filter_items_by_weather(filtered_items, hot)
    if not weather_filtered_items:
        logging.warning(f"No items found suitable for {'hot' if hot else 'cold'} weather.")
        return []

    # 2. Pick an upper body item to set the base color scheme
    upper = pick_upper_body(weather_filtered_items, desired_aesthetic, base_colors=[])
    if not upper:
        logging.warning("No upper body garment found for outfit.")
        return []

    base_colors = upper.get("color", [])

    # 3. Pick other items that are compatible with the upper body item (all use fallback logic)
    lower = pick_lower_body(weather_filtered_items, hot, desired_aesthetic, base_colors)
    if not lower:
        logging.warning("No lower body garment found for outfit.")
        return []

    footwear = pick_footwear(weather_filtered_items, desired_aesthetic, base_colors)
    if not footwear:
        logging.warning("No footwear found for outfit.")
        return []

    outfit = [upper]

    # 4. Add outerwear for cold weather (also uses 3-tier fallback logic)
    if not hot:
        outer = pick_outerwear(weather_filtered_items, desired_aesthetic, base_colors)
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