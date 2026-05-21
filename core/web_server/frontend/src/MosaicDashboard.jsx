// MosaicDashboard.jsx — at-a-glance view of a mosaic, one row per
// registered pipeline. Each row shows a thumbnail per filter; clicking
// a thumbnail opens that pipeline's output in a full-screen viewer.
//
// Data source: `/api/pipeline_outputs/mosaic/<id>` returns one entry
// per pipeline (PixInsight Master, PixInsight Downscaled, Subframe,
// + any user-registered pipelines). The (implementation, quality)
// fields on each pipeline entry are still threaded into thumbnail
// URLs because the on-disk artifacts are still keyed that way; the
// pipeline name is the user-facing label.
import React, { useContext, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Box, Paper, Typography, Divider } from "@mui/material";

import AstroFlowContext from "./AstroFlowContext.jsx";


function VersionThumbnail({
    mosaicId, implementation, quality, filter, exists, mtime,
}) {
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
                    alt={`${filter}`}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
            ) : (
                <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    not yet built
                </Typography>
            )}
            <Box sx={{
                position: 'absolute', bottom: 0, left: 0, right: 0,
                backgroundColor: 'rgba(0,0,0,0.6)',
                color: 'white',
                fontSize: 12,
                px: 1, py: 0.25,
            }}>
                {filter || '—'}
            </Box>
        </Box>
    );
}


function PipelineSection({ mosaicId, pipeline, filters }) {
    // outputs[i] aligns with filters[i] (backend guarantees same order).
    const byFilter = {};
    for (let i = 0; i < filters.length; i++) {
        byFilter[filters[i]] = pipeline.outputs[i];
    }
    return (
        <Box sx={{ mb: 4 }}>
            <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                {pipeline.name}
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Box sx={{ display: 'flex', gap: 1, overflowX: 'auto', flex: 1 }}>
                {filters.map((f) => {
                    const o = byFilter[f] || { filter: f, exists: false };
                    return (
                        <VersionThumbnail
                            key={f}
                            mosaicId={mosaicId}
                            implementation={pipeline.implementation}
                            quality={pipeline.quality}
                            filter={f}
                            exists={o.exists}
                            mtime={o.mtime}
                        />
                    );
                })}
            </Box>
        </Box>
    );
}


export default function MosaicDashboard() {
    const { mosaic_id } = useParams();
    const { mosaics, computeGraphStatus } = useContext(AstroFlowContext);
    const [data, setData] = useState(null);

    const mosaic = mosaics?.[mosaic_id];

    useEffect(() => {
        fetch(`/api/pipeline_outputs/mosaic/${mosaic_id}`)
            .then((res) => res.ok ? res.json() : null)
            .then(setData)
            .catch(() => setData(null));
    }, [mosaic_id, computeGraphStatus]);

    if (!mosaic) {
        return <Typography sx={{ p: 3 }}>Loading…</Typography>;
    }

    const filters = data?.filters || [];
    const pipelines = data?.pipelines || [];

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

            {pipelines.length === 0 ? (
                <Typography sx={{ color: 'text.secondary' }}>
                    No pipelines registered.
                </Typography>
            ) : (
                pipelines.map((p) => (
                    <PipelineSection
                        key={p.name}
                        mosaicId={mosaic_id}
                        pipeline={p}
                        filters={filters}
                    />
                ))
            )}
        </Paper>
    );
}
