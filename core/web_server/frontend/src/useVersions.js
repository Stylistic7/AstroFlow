import { useEffect, useState } from "react";


// Default tap-point display label per object-type. Used when a viewer
// URL carries a `:pipeline` segment but no `:display` (the user
// clicked a thumbnail; we fill in the most prominent tap point).
// Pipelines emit tap names freely via `@makes_image(display="X")`;
// the dashboard surfaces every emitted name, so these defaults only
// matter for code paths that need to construct a URL from scratch.
export const DEFAULT_DISPLAY_FOR_TYPE = {
    mosaic: "Mosaic",
    stack: "Master",
    target: "Channel",
    calibration_stack: "Master",
};


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


// Fetch the registered pipelines from /api/pipelines. Cached per session.
export function usePipelines() {
    const [pipelines, setPipelines] = useState([]);

    useEffect(() => {
        let cancelled = false;
        fetch("/api/pipelines")
            .then((res) => (res.ok ? res.json() : null))
            .then((d) => {
                if (!cancelled && d) setPipelines(d.pipelines || []);
            })
            .catch((e) => console.warn(e));
        return () => {
            cancelled = true;
        };
    }, []);

    return pipelines;
}


// Pipelines the user can pick: every registered pipeline. (Legacy
// fallback to on-disk-only names from the version matrix kept for
// the transition; user-defined pipelines surface via the registry.)
export function pipelineOptions(versions, availablePipelines) {
    return availablePipelines.map((p) => p.name);
}
