// MosaicDashboard.jsx — at-a-glance view of a mosaic. One section per
// registered pipeline, one ROW per tap point within that pipeline,
// thumbnails per filter inside each row.
//
// Data source: `/api/pipeline_outputs/mosaic/<id>` returns one entry
// per pipeline, with that pipeline's tap points (display labels) and
// per-(tap, filter) on-disk existence. Tap points are discovered at
// pipeline-trace time from `@makes_image(display=...)` annotations,
// so user pipelines surface in this dashboard automatically.
import React, { useContext, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Box, Paper, Typography, Divider } from "@mui/material";

import AstroFlowContext from "./AstroFlowContext.jsx";


function VersionThumbnail({
    objectType, objectId, pipeline, display, filter, exists, mtime,
}) {
    const navigate = useNavigate();
    // Tap-keyed URLs: `(pipeline, display)` together identify the
    // artifact within the pipeline's graph. `<path:>` on the backend
    // lets the slash inside the joined segment pass through.
    const tap = `${encodeURIComponent(pipeline)}/${encodeURIComponent(display)}`;
    const url = exists
        ? `/tap/thumbnail/${objectType}/${objectId}/${tap}/${filter}?v=${mtime}`
        : null;
    const href = `/${objectType}/${objectId}/${tap}/${filter}`;

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
                    alt={`${display} — ${filter}`}
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


function PipelineSection({ objectType, objectId, pipeline, filters }) {
    return (
        <Box sx={{ mb: 4 }}>
            <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                {pipeline.name}
            </Typography>
            <Divider sx={{ mb: 2 }} />
            {pipeline.taps.map((tap) => {
                // outputs[i] aligns with filters[i] (backend
                // guarantees same order within a tap).
                const byFilter = {};
                for (let i = 0; i < filters.length; i++) {
                    byFilter[filters[i]] = tap.outputs[i];
                }
                return (
                    <Box key={tap.display} sx={{ mb: 2 }}>
                        <Typography variant="body2" sx={{
                            color: 'text.secondary',
                            mb: 0.5,
                        }}>
                            {tap.display}
                        </Typography>
                        <Box sx={{
                            display: 'flex', gap: 1, overflowX: 'auto', flex: 1,
                        }}>
                            {filters.map((f) => {
                                const o = byFilter[f] || { filter: f, exists: false };
                                return (
                                    <VersionThumbnail
                                        key={f}
                                        objectType={objectType}
                                        objectId={objectId}
                                        pipeline={pipeline.name}
                                        display={tap.display}
                                        filter={f}
                                        exists={o.exists}
                                        mtime={o.mtime}
                                    />
                                );
                            })}
                        </Box>
                    </Box>
                );
            })}
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
                        objectType="mosaic"
                        objectId={mosaic_id}
                        pipeline={p}
                        filters={filters}
                    />
                ))
            )}
        </Paper>
    );
}
