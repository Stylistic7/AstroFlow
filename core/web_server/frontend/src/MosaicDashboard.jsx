// MosaicDashboard.jsx — at-a-glance view of a mosaic: what each pipeline
// PRODUCED, and whether it is current. One section per pipeline, showing:
//
//   EVERY FINISH the pipeline lists — composites and single channels alike —
//   each with its build status, deep-linked into the viewer. (A target's
//   page also lists each pipeline's stacked masters; see PipelineSection.)
//
// Data source: `/api/finishes/mosaic/<id>` returns one entry per pipeline,
// each with its finishes in preference order (`absent` = traced but not
// built yet). Finishes are discovered at pipeline-trace time, so user
// pipelines and user-authored finishes surface here automatically.
import React, { useContext, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Box, Paper, Typography, Divider, Button } from "@mui/material";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";

import AstroFlowContext from "./AstroFlowContext.jsx";
import DiscardResultsDialog from "./DiscardResultsDialog.jsx";
import { buildUrl, buildApiUrl } from "./urls.js";
import { useFinishes } from "./useVersions.js";
import StackThumbnail from "./StackThumbnail.jsx";


// A finish's preview tile: an image (or "not yet built"), its name, an OUT OF
// DATE chip; the whole tile opens `href`. (Masters are StackThumbnails, which
// carry the stack's own filter, subs, hours and exposure widget.)
function Tile({ src, caption, status, href }) {
    const navigate = useNavigate();
    return (
        <Box
            onClick={() => navigate(href)}
            sx={{
                minWidth: 0,
                aspectRatio: '3 / 2',
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'divider',
                overflow: 'hidden',
                position: 'relative',
                // Always clickable — even before it is built, the viewer page
                // is useful for the workflow graph + progress.
                cursor: 'pointer',
                backgroundColor: src ? 'transparent' : 'rgba(255,255,255,0.04)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
            }}
        >
            {src ? (
                <img src={src} alt={caption}
                     style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
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
                {caption}
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


const tileGrid = {
    display: 'grid', gap: 1.5, width: '100%',
    gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
};


function GroupTitle({ children }) {
    return (
        <Typography variant="subtitle2" sx={{ color: 'text.secondary', mt: 1, mb: 1 }}>
            {children}
        </Typography>
    );
}


// One section per pipeline, in two halves: what the PIPELINE made (a
// target's stacked masters, `pipeline.masters`, each opening the stack viewer
// in this pipeline) and the FINISHES applied to its outputs, each opening the
// finish viewer. No hero image: the full-screen viewer is the one big picture
// (2026-10-17). Exported: TargetPanel and MosaicDashboard both render it; a
// mosaic's pipelines list finishes only.
export function PipelineSection({ objectType, objectId, pipeline, hierarchy, masterBadges }) {
    const [discardOpen, setDiscardOpen] = useState(false);
    const own = [{ type: objectType, id: objectId }];
    const pageHierarchy = hierarchy || own;
    const masters = pipeline.masters || [];
    return (
        <Box sx={{ mb: 5 }}>
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
            <Divider sx={{ mb: 1 }} />
            <DiscardResultsDialog
                objectType={objectType}
                objectId={objectId}
                pipeline={pipeline.name}
                open={discardOpen}
                onClose={() => setDiscardOpen(false)}
            />

            {masters.length > 0 && (
                <>
                    <GroupTitle>Pipeline outputs · stacked masters</GroupTitle>
                    <Box sx={{ ...tileGrid, mb: 2 }}>
                        {masters.map((m) => {
                            const leaf = [{ type: 'stack', id: m.stack_id }];
                            const view = m.finish && { pipeline: pipeline.name, finish: m.finish };
                            return (
                                <StackThumbnail
                                    key={m.stack_id}
                                    stack_id={m.stack_id}
                                    src={view && m.status !== 'absent'
                                        ? `${buildApiUrl('/tap/thumbnail', leaf, view)}?v=${m.mtime}` : undefined}
                                    status={m.status}
                                    handleClick={() => window.open(
                                        buildUrl({ hierarchy: [...pageHierarchy, ...leaf], view }), '_blank')}
                                >
                                    {masterBadges && masterBadges(m.stack_id)}
                                </StackThumbnail>
                            );
                        })}
                    </Box>
                </>
            )}

            {pipeline.finishes.length > 0 && (
                <>
                    {masters.length > 0 && <GroupTitle>Finishes</GroupTitle>}
                    {/* Every finish this pipeline lists, in preference order. */}
                    <Box sx={tileGrid}>
                        {pipeline.finishes.map((f) => {
                            const view = { pipeline: pipeline.name, finish: f.name };
                            return (
                                <Tile
                                    key={f.name}
                                    src={f.status !== 'absent'
                                        ? `${buildApiUrl('/tap/thumbnail', own, view)}?v=${f.mtime}` : null}
                                    caption={f.name}
                                    status={f.status}
                                    href={buildUrl({ hierarchy: pageHierarchy, view })}
                                />
                            );
                        })}
                    </Box>
                </>
            )}
        </Box>
    );
}


export default function MosaicDashboard() {
    const { mosaic_id } = useParams();
    const { mosaics, computeGraphStatus } = useContext(AstroFlowContext);
    const data = useFinishes("mosaic", mosaic_id, computeGraphStatus);

    const mosaic = mosaics?.[mosaic_id];

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
                    Nothing produced for this mosaic yet — the compute graph is still working on it.
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
