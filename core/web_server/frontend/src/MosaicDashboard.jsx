// MosaicDashboard.jsx — at-a-glance view of a mosaic across every
// (implementation × quality level × filter). Click a thumbnail to open
// that specific version in a full-screen viewer.
import React, { useContext, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Box, Paper, Typography, Divider } from "@mui/material";

import AstroFlowContext from "./AstroFlowContext.jsx";

// Stable quality-level ordering: best → worst, same as backend.
const QUALITY_ORDER = [
    "PostProcessed",
    "Master",
    "DownscaledMaster",
    "Subframe",
];

const QUALITY_LABEL = {
    PostProcessed: "Post-Processed",
    Master: "Master",
    DownscaledMaster: "Downscaled",
    Subframe: "Subframe",
};


function VersionThumbnail({ mosaicId, implementation, quality, filter, exists, mtime }) {
    const navigate = useNavigate();
    const url = exists
        ? `/thumbnail/mosaic/${mosaicId}/${implementation}/${quality}/${filter}?v=${mtime}`
        : null;
    const href = `/mosaic/${mosaicId}/${implementation}/${quality}/${filter}`;

    const sx = {
        width: 200,
        height: 150,
        flexShrink: 0,
        borderRadius: 1,
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
        position: 'relative',
        cursor: exists ? 'pointer' : 'default',
        backgroundColor: exists ? 'transparent' : 'rgba(255,255,255,0.04)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    };

    return (
        <Box sx={sx} onClick={exists ? () => navigate(href) : undefined}>
            {url ? (
                <img
                    src={url}
                    alt={`${filter} — ${implementation} ${quality}`}
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                />
            ) : (
                <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    {filter} — not yet
                </Typography>
            )}
            <Box sx={{
                position: 'absolute', bottom: 0, left: 0, right: 0,
                backgroundColor: 'rgba(0,0,0,0.6)',
                color: 'white',
                fontSize: 12,
                px: 1, py: 0.25,
            }}>
                {filter}
            </Box>
        </Box>
    );
}


function QualityRow({ mosaicId, implementation, quality, filters, versions }) {
    // Map (filter) → version for fast lookup.
    const byFilter = {};
    for (const v of versions) {
        if (v.implementation === implementation && v.quality === quality) {
            byFilter[v.filter] = v;
        }
    }
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <Typography variant="body2" sx={{ width: 140, color: 'text.secondary' }}>
                {QUALITY_LABEL[quality] || quality}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, overflowX: 'auto', flex: 1 }}>
                {filters.map((f) => {
                    const v = byFilter[f] || { implementation, quality, filter: f, exists: false };
                    return (
                        <VersionThumbnail
                            key={`${f}`}
                            mosaicId={mosaicId}
                            {...v}
                        />
                    );
                })}
            </Box>
        </Box>
    );
}


function ImplementationSection({ mosaicId, implementation, filters, versions }) {
    return (
        <Box sx={{ mb: 4 }}>
            <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                {implementation}
            </Typography>
            <Divider sx={{ mb: 2 }} />
            {QUALITY_ORDER.map((q) => (
                <QualityRow
                    key={q}
                    mosaicId={mosaicId}
                    implementation={implementation}
                    quality={q}
                    filters={filters}
                    versions={versions}
                />
            ))}
        </Box>
    );
}


export default function MosaicDashboard() {
    const { mosaic_id } = useParams();
    const { mosaics, computeGraphStatus } = useContext(AstroFlowContext);
    const [data, setData] = useState(null);

    const mosaic = mosaics?.[mosaic_id];

    useEffect(() => {
        fetch(`/api/mosaic_versions/${mosaic_id}`)
            .then((res) => res.ok ? res.json() : null)
            .then(setData)
            .catch(() => setData(null));
    }, [mosaic_id, computeGraphStatus]);

    if (!mosaic) {
        return <Typography sx={{ p: 3 }}>Loading…</Typography>;
    }

    const filters = data?.filters || [];
    const versions = data?.versions || [];

    // Unique implementations, in registry order (PixInsight → AstroFlow → OSS).
    const implementations = [];
    const seen = new Set();
    for (const v of versions) {
        if (!seen.has(v.implementation)) {
            seen.add(v.implementation);
            implementations.push(v.implementation);
        }
    }

    return (
        <Paper elevation={2} sx={{
            borderRadius: 0,
            borderTop: "1px solid",
            borderColor: "divider",
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
            minHeight: 0,
            p: 3,
            overflow: 'auto',
        }}>
            <Typography variant="h5" sx={{ mb: 3 }}>
                {mosaic.name || `Mosaic ${mosaic_id.slice(0, 8)}`}
            </Typography>

            {implementations.length === 0 ? (
                <Typography sx={{ color: 'text.secondary' }}>
                    No versions produced yet.
                </Typography>
            ) : (
                implementations.map((impl) => (
                    <ImplementationSection
                        key={impl}
                        mosaicId={mosaic_id}
                        implementation={impl}
                        filters={filters}
                        versions={versions}
                    />
                ))
            )}
        </Paper>
    );
}
