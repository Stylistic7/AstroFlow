"""Clear the persisted shopping cart.

Usage: `python -m scripts.clear_shopping_cart` (from the `core/`
working dir).

Opens the diskcache directly — does NOT need Ray running. Useful
when the cart shape changes (e.g. the QualityLevel → pipeline
collapse, which leaves any pre-existing 4-tuple entries unable to
parse cleanly) and you want a clean slate without nuking the rest
of the cache.
"""
from context.cache import _create_cache


def main() -> None:
    cache = _create_cache()
    existing = cache.get("shopping_cart") or []
    if not existing:
        print("Shopping cart is already empty.")
        return
    print(f"Clearing {len(existing)} cart entries:")
    for entry in existing:
        print(f"  {entry!r}")
    if "shopping_cart" in cache:
        del cache["shopping_cart"]
    print("Cleared.")


if __name__ == "__main__":
    main()
