/**
 * Canonical-preview <img> that surfaces the product's freshness.
 *
 * The preview endpoint serves last-good generations while a rebuild is in
 * flight (graceful N/N-1) and reports which it served in the
 * `X-Product-Status` response header. A plain <img src> cannot read
 * headers, so this component fetches the PNG, shows it via a blob URL, and
 * overlays the same "OUT OF DATE" chip the tile viewer uses when the
 * header says so. Before 2026-08-31 every preview consumer rendered stale
 * composites indistinguishably from current ones.
 */
import React, { useEffect, useState } from 'react';
import Box from '@mui/material/Box';

export default function PreviewImage({ src, alt, style, onAbsent }) {
    const [blobUrl, setBlobUrl] = useState(null);
    const [status, setStatus] = useState(null);

    useEffect(() => {
        let revoked = null;
        let cancelled = false;
        setBlobUrl(null);
        setStatus(null);
        fetch(src)
            .then(async (r) => {
                if (!r.ok) throw new Error(String(r.status));
                const s = r.headers.get('X-Product-Status');
                const blob = await r.blob();
                if (cancelled) return;
                revoked = URL.createObjectURL(blob);
                setBlobUrl(revoked);
                setStatus(s);
            })
            .catch(() => { if (!cancelled) onAbsent?.(); });
        return () => {
            cancelled = true;
            if (revoked) URL.revokeObjectURL(revoked);
        };
    }, [src]);

    if (!blobUrl) return null;
    return (
        <Box sx={{ position: 'relative', display: 'inline-block',
                   maxWidth: '100%' }}>
            <img src={blobUrl} alt={alt} style={style} />
            {status === 'out_of_date' && (
                <Box sx={{
                    position: 'absolute', top: 8, left: 8,
                    backgroundColor: 'warning.main',
                    color: 'warning.contrastText',
                    px: 1, py: 0.25, borderRadius: 1,
                    fontSize: 11, fontWeight: 700, letterSpacing: 0.4,
                    pointerEvents: 'none',
                }}>
                    OUT OF DATE
                </Box>
            )}
        </Box>
    );
}
