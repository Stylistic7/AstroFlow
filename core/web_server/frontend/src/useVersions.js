import { useEffect, useState } from "react";

// Display order (best → worst). Mirrors backend's QUALITY_PRIORITY in
// previews/types.py; kept in sync by convention. The constant lives here
// so consumers (breadcrumb pills, version cycling) share one ordering.
export const QUALITY_ORDER = [
    "PostProcessed",
    "Master",
    "DownscaledMaster",
    "Subframe",
];


// One fetcher for /api/object_versions. Every viewer (mosaic, stack, …)
// reads from this hook so there's a single source of truth for "what
// versions exist for this object." URL params are the single source of
// truth for "what version is currently being viewed" — this hook never
// stores selection state.
//
// `refreshKey` is an opaque value that, when changed, forces a re-fetch.
// Pass `computeGraphStatus` (or any SSE-driven counter) so the pill
// crumbs light up live as new versions finish producing.
export function useVersionsMatrix(objectType, objectId, refreshKey) {
    const [data, setData] = useState({ filters: [], versions: [] });

    useEffect(() => {
        if (!objectType || !objectId) return;
        let cancelled = false;
        fetch(`/api/object_versions/${objectType}/${objectId}`)
            .then((res) => (res.ok ? res.json() : null))
            .then((d) => {
                if (!cancelled && d) setData(d);
            })
            .catch((e) => console.warn(e));
        return () => {
            cancelled = true;
        };
    }, [objectType, objectId, refreshKey]);

    return data;
}


export function versionExists(versions, impl, quality, filter) {
    return versions.some(
        (v) =>
            v.implementation === impl &&
            v.quality === quality &&
            v.filter === filter &&
            v.exists
    );
}


export function implementationsFrom(versions) {
    return [...new Set(versions.map((v) => v.implementation))];
}


// Qualities (in priority order) that exist for the current (impl, filter).
export function qualitiesFor(versions, impl, filter) {
    return QUALITY_ORDER.filter((q) => versionExists(versions, impl, q, filter));
}


function cycle(list, current, dir) {
    const idx = list.indexOf(current);
    if (idx < 0) return current;
    return list[(idx + dir + list.length) % list.length];
}


// Step through `list` from `current` in `dir`, returning the first
// candidate for which `predicate(candidate)` is true. Bails after one
// full lap so non-existent versions don't trap us in an infinite walk.
export function cycleToExisting(list, current, dir, predicate) {
    let candidate = current;
    for (let i = 0; i < list.length; i++) {
        candidate = cycle(list, candidate, dir);
        if (predicate(candidate)) return candidate;
    }
    return current;
}


// Keyboard shortcuts for stepping between versions while staying on the
// viewer page:  [ ]  implementation,  { }  quality,  , .  filter.
// The caller owns URL construction by supplying `onChange`, which
// receives the next `{implementation, quality, filter}` triple and is
// expected to `navigate(...)` accordingly. Filter cycling is no-op when
// the route has no filter (current.filter is null/undefined).
export function useVersionCycling({ data, current, onChange }) {
    const { implementation, quality, filter } = current;

    useEffect(() => {
        const handler = (e) => {
            if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

            const impls = implementationsFrom(data.versions);
            const quals = qualitiesFor(data.versions, implementation, filter);
            const filters = data.filters;

            if (e.key === "[" || e.key === "]") {
                const next = cycleToExisting(
                    impls, implementation, e.key === "]" ? 1 : -1,
                    (i) => versionExists(data.versions, i, quality, filter),
                );
                if (next !== implementation) onChange({ ...current, implementation: next });
                e.preventDefault();
            } else if (e.key === "{" || e.key === "}") {
                const next = cycleToExisting(
                    quals, quality, e.key === "}" ? 1 : -1,
                    (q) => versionExists(data.versions, implementation, q, filter),
                );
                if (next !== quality) onChange({ ...current, quality: next });
                e.preventDefault();
            } else if ((e.key === "," || e.key === ".") && filter != null) {
                const next = cycleToExisting(
                    filters, filter, e.key === "." ? 1 : -1,
                    (f) => versionExists(data.versions, implementation, quality, f),
                );
                if (next !== filter) onChange({ ...current, filter: next });
                e.preventDefault();
            }
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, [data, implementation, quality, filter, onChange]);
}
