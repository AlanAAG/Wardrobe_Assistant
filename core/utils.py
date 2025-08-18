from typing import List, Dict

def categorize_items_by_category(items: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorizes a list of item dictionaries by their 'category' field."""
    categorized: Dict[str, List[Dict]] = {}
    for item in items or []:
        category = item.get("category", "Unknown")
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(item)
    return categorized
