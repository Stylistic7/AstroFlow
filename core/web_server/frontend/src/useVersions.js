import { useEffect, useState } from "react";


// One fetcher for the per-object pipeline / tap matrix — the ONLY reader of
// /api/pipeline_outputs. The endpoint is the single source of truth for
// "which pipelines produced which taps for this object" AND the default
// landing view (`.default = {pipeline, display, filter}`; the backend owns
// the pipeline/display preference, see catalog.default_product_for). URL
// params remain the single source of truth for "what's currently being
// viewed"; this hook never stores selection state.
//
// ONE fetch per (object, refreshKey), however many components ask. The
// breadcrumb pills, the page's redirect-to-default and the viewer's finish
// picker all read the matrix of the same object; when each held its own
// copy they populated at different moments (and one of them, missing a
// parameter, never populated at all) — so the pills could disagree with
// the image on the same screen. Every subscriber now sees the same payload
// at the same instant, and a live SSE tick refetches it once.
//
// Returns `null` until the first response lands (callers treat null as
// "still resolving"); keeps the previous payload across refetches so live
// ticks don't flash the UI empty.
const matrices = new Map();   // "type/id" → { data, refreshKey, listeners }

function fetchMatrix(entry, key, refreshKey) {
    entry.refreshKey = refreshKey;
    fetch(`/api/pipeline_outputs/${key}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((d) => {
            // A refetch was issued meanwhile — that one's answer wins.
            if (!d || entry.refreshKey !== refreshKey) return;
            entry.data = d;
            entry.listeners.forEach((fn) => fn(d));
        })
        .catch((e) => console.warn(e));
}

export function useVersionsMatrix(objectType, objectId, refreshKey) {
    const key = objectType && objectId ? `${objectType}/${objectId}` : null;
    const [data, setData] = useState(() => (key && matrices.get(key)?.data) ?? null);

    useEffect(() => {
        if (!key) return undefined;
        let entry = matrices.get(key);
        if (!entry) {
            entry = { data: null, refreshKey: undefined, listeners: new Set() };
            matrices.set(key, entry);
        }
        entry.listeners.add(setData);
        setData(entry.data);
        if (entry.refreshKey !== refreshKey) fetchMatrix(entry, key, refreshKey);
        return () => { entry.listeners.delete(setData); };
    }, [key, refreshKey]);

    return data;
}


// Fetch the registered pipelines from /api/pipelines — what the CART can
// queue (a superset of what any one object has produced). Cached per session.
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
