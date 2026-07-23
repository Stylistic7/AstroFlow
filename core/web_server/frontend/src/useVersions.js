import { useEffect, useState } from "react";


// One fetcher for the per-object pipeline / tap matrix (the ONLY hook for
// /api/pipeline_outputs — usePipelineOutputs.js was a duplicate and is gone).
// The endpoint is the single source of truth for "what artifacts has each
// registered pipeline produced for this object" AND the default landing view
// (`.default = {pipeline, display, filter}` — the backend owns the pipeline/
// display preference now, see catalog.default_product_for). URL params remain
// the single source of truth for "what's currently being viewed"; this hook
// never stores selection state.
//
// Returns `null` until the first response lands (callers treat null as "still
// resolving"); keeps the previous payload across refetches so live SSE ticks
// don't flash the UI empty.
//
// `refreshKey` is an opaque value that, when changed, forces a re-fetch.
// Pass `computeGraphStatus` (or any SSE-driven counter) so the
// breadcrumb pills light up live as new outputs finish producing.
export function useVersionsMatrix(objectType, objectId, refreshKey, ancestors = "") {
    const [data, setData] = useState(null);

    useEffect(() => {
        if (!objectType || !objectId) return;
        let cancelled = false;
        // A child leaf (a stack under a target) has no snapshot of its own; the
        // ancestors let the backend resolve its taps from the cart-item ancestor.
        const qs = ancestors ? `?ancestors=${encodeURIComponent(ancestors)}` : "";
        fetch(`/api/pipeline_outputs/${objectType}/${objectId}${qs}`)
            .then((res) => (res.ok ? res.json() : null))
            .then((d) => {
                if (!cancelled && d) setData(d);
            })
            .catch((e) => console.warn(e));
        return () => {
            cancelled = true;
        };
    }, [objectType, objectId, refreshKey, ancestors]);

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
