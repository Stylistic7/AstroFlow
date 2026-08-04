// MosaicDashboard.jsx — at-a-glance view of a mosaic. One section per
// registered pipeline, one ROW per tap point within that pipeline (in ORDER OF
// COMPUTATION), showing ONLY the filters that actually route through that tap.
//
// A tap node isn't filter-specific — its real filter set is whatever the
// pipeline×object trace produced for it — so each row renders its own
// `tap.outputs` (no global filter grid, no phantom blanks). The images in a row
// split the full viewport width, so a one-filter tap (e.g. a Hubble composite)
// renders large and a many-filter tap (e.g. Linear H/S/O) renders as a strip.
//
// Data source: `/api/pipeline_outputs/mosaic/<id>` returns one entry per
// pipeline, each with its tap points (display labels, in trace order) and, per
// tap, the real (filter, on-disk-existence) outputs. Tap points are discovered
// at pipeline-trace time from `@makes_image(display=...)` annotations, so user
// pipelines surface in this dashboard automatically.
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
        // Fill an equal share of the row's width. With N siblings each `flex:1`,
        // every image is 1/N of the viewport — so a one-filter row renders large
        // and a many-filter row renders small, all filling the width edge-to-edge.
        flex: '1 1 0',
        minWidth: 0,
        aspectRatio: '3 / 2',
        maxHeight: '65vh',
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


function PipelineSection({ objectType, objectId, pipeline }) {
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
            {pipeline.taps.map((tap) => (
                <Box key={tap.display} sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{
                        color: 'text.secondary',
                        mb: 0.5,
                    }}>
                        {tap.display}
                    </Typography>
                    {/* One row per tap, showing only the filters that route
                        through it. The images share the full width (each
                        flex:1), so a one-filter tap renders large and a
                        many-filter tap renders as a strip. */}
                    <Box sx={{
                        display: 'flex', gap: 1, width: '100%',
                        alignItems: 'flex-start',
                    }}>
                        {tap.outputs.map((o) => (
                            <VersionThumbnail
                                key={o.filter ?? '—'}
                                objectType={objectType}
                                objectId={objectId}
                                pipeline={pipeline.name}
                                display={tap.display}
                                filter={o.filter}
                                exists={o.exists}
                                mtime={o.mtime}
                                status={o.status}
                            />
                        ))}
                    </Box>
                </Box>
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
        fetch(`/api/pipeline_outputs/mosaic/${mosaic_id}`)
            .then((res) => res.ok ? res.json() : null)
            .then(setData)
            .catch(() => setData(null));
    }, [mosaic_id, computeGraphStatus]);

    if (!mosaic) {
        return <Typography sx={{ p: 3 }}>Loading…</Typography>;
    }

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
                    />
                ))
            )}
        </Paper>
    );
}
