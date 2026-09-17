// Filter ROLES (the single letters that name single-channel finishes and
// the `role` on /api/stacks entries) → human-readable names. Anywhere a
// role is shown to a user, route through `filterLabel(role)` so the
// convention stays in one place. URLs and stored objects keep the
// single-letter form; a composite finish name falls through unchanged.
const FILTER_LABELS = {
    L: "Luminance",
    R: "Red",
    G: "Green",
    B: "Blue",
    H: "Hydrogen-α",
    S: "Sulfur II",
    O: "Oxygen III",
};


export function filterLabel(id) {
    if (id == null) return id;
    return FILTER_LABELS[id] ?? id;
}
