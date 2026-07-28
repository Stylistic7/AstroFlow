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
import { Box, Paper, Typography, Divider, Button } from "@mui/material";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";

import AstroFlowContext from "./AstroFlowContext.jsx";
import DiscardResultsDialog from "./DiscardResultsDialog.jsx";
import { buildUrl, buildApiUrl } from "./urls.js";


function VersionThumbnail({
    objectType, objectId, pipeline, display, filter, exists, mtime, status,
}) {
    const navigate = useNavigate();
    const hierarchy = [{ type: objectType, id: objectId }];
    const view = { pipeline, display, filter };
    const url = exists
        ? `${buildApiUrl('/tap/thumbnail', hierarchy, view)}?v=${mtime}`
        : null;
    const href = buildUrl({ hierarchy, view });

    const sx = {
        width: 200,
        height: 150,
        flexShrink: 0,
        borderRadius: 1,
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
        position: 'relative',
        // Always clickable — even when the file isn't yet on disk,
        // the per-(pipeline, display, filter) viewer page is useful
        // for inspecting the workflow graph + tracking progress.
        cursor: 'pointer',
        backgroundColor: exists ? 'transparent' : 'rgba(255,255,255,0.04)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    };

    return (
        <Box sx={sx} onClick={() => navigate(href)}>
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
            {status === 'out_of_date' && (
                <Box sx={{
                    position: 'absolute', top: 4, right: 4,
                    backgroundColor: 'warning.main',
                    color: 'warning.contrastText',
                    fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
                    px: 0.5, py: 0.25, borderRadius: 0.5, lineHeight: 1.3,
                }}>
                    OUT OF DATE
                </Box>
            )}
        </Box>
    );
}


function PipelineSection({ objectType, objectId, pipeline, filters }) {
    const [discardOpen, setDiscardOpen] = useState(false);
    return (
        <Box sx={{ mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="h6" sx={{ color: 'primary.main' }}>
                    {pipeline.name}
                </Typography>
                <Box sx={{ flexGrow: 1 }} />
                <Button
                    size="small"
                    color="warning"
                    startIcon={<DeleteSweepIcon />}
                    onClick={() => setDiscardOpen(true)}
                >
                    Discard results…
                </Button>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <DiscardResultsDialog
                objectType={objectType}
                objectId={objectId}
                pipeline={pipeline.name}
                open={discardOpen}
                onClose={() => setDiscardOpen(false)}
            />
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
                                const o = byFilter[f] || { filter: f, exists: false, status: 'absent' };
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
                                        status={o.status}
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
