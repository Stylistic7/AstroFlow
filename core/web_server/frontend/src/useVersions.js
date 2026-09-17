import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { buildUrl } from "./urls.js";


// One fetcher for the per-object pipeline / finish list — the ONLY reader of
// /api/finishes. The endpoint is the single source of truth for "which
// pipelines produced which finishes for this object" AND the default
// landing view (`.default = {pipeline, finish}`; the backend owns the
// preference). URL params remain the single source of truth for "what's
// currently being viewed"; this hook never stores selection state.
//
// ONE fetch per (object, refreshKey), however many components ask. The
// breadcrumb pills, the page's redirect-to-default and the viewer all read
// the list of the same object; when each held its own copy they populated
// at different moments — so the pills could disagree with the image on the
// same screen. Every subscriber now sees the same payload at the same
// instant, and a live SSE tick refetches it once.
//
// Returns `null` until the first response lands (callers treat null as
// "still resolving"); keeps the previous payload across refetches so live
// ticks don't flash the UI empty.
const lists = new Map();   // "type/id" → { data, refreshKey, listeners }

function fetchList(entry, key, refreshKey) {
    entry.refreshKey = refreshKey;
    fetch(`/api/finishes/${key}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((d) => {
            // A refetch was issued meanwhile — that one's answer wins.
            if (!d || entry.refreshKey !== refreshKey) return;
            entry.data = d;
            entry.listeners.forEach((fn) => fn(d));
        })
        .catch((e) => console.warn(e));
}

export function useFinishes(objectType, objectId, refreshKey) {
    const key = objectType && objectId ? `${objectType}/${objectId}` : null;
    const [data, setData] = useState(() => (key && lists.get(key)?.data) ?? null);

    useEffect(() => {
        if (!key) return undefined;
        let entry = lists.get(key);
        if (!entry) {
            entry = { data: null, refreshKey: undefined, listeners: new Set() };
            lists.set(key, entry);
        }
        entry.listeners.add(setData);
        setData(entry.data);
        if (entry.refreshKey !== refreshKey) fetchList(entry, key, refreshKey);
        return () => { entry.listeners.delete(setData); };
    }, [key, refreshKey]);

    return data;
}


// The URL might name a finish this pipeline doesn't list — typically a view
// carried across from another object or pipeline. Redirect to the pipeline's
// first finish so the page doesn't 404-loop on every tile-metadata fetch.
export function useValidFinish(finishes, hierarchy, pipeline, finish) {
    const navigate = useNavigate();
    const current = finishes?.pipelines.find((p) => p.name === pipeline);
    const replacement = current && !current.finishes.some((f) => f.name === finish)
        ? current.finishes[0]?.name
        : null;
    const to = replacement
        ? buildUrl({ hierarchy, view: { pipeline, finish: replacement } })
        : null;
    useEffect(() => {
        if (to) navigate(to, { replace: true });
    }, [to, navigate]);
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
