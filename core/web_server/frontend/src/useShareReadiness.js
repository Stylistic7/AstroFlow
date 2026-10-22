// Whether a stack's master may go into a shared master folder, with the
// reasons (`GET /api/share_readiness/stack/<id>`, metadata/share_gate.py):
// `{ ready, blocking: [{code, title, detail}], warnings: [...] }`, or null
// while loading. One fetch per stack per page load, shared by the card and
// the sharing menu.
import { useEffect, useState } from 'react';

const cache = new Map();

export default function useShareReadiness(stackId) {
  const [readiness, setReadiness] = useState(cache.get(stackId) || null);
  useEffect(() => {
    if (!stackId) return;
    if (cache.has(stackId)) { setReadiness(cache.get(stackId)); return; }
    let live = true;
    fetch(`/api/share_readiness/stack/${stackId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((v) => { if (v) cache.set(stackId, v); if (live) setReadiness(v); })
      .catch(() => { if (live) setReadiness(null); });
    return () => { live = false; };
  }, [stackId]);
  return readiness;
}

export function forgetShareReadiness(stackId) {
  cache.delete(stackId);
}
