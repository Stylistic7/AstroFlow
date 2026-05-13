// Single-letter filter IDs (matching the codebase and URLs) → human-
// readable names used for display. Anywhere a filter is shown to a
// user, route through `filterLabel(id)` so the convention stays in one
// place. URLs and stored objects keep the single-letter form.
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
