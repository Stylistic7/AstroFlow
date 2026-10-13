// MosaicDashboard.jsx — at-a-glance view of a mosaic: what each pipeline
// PRODUCED, and whether it is current. One section per pipeline, showing:
//
//   1. THE DELIVERABLE — that pipeline's preferred finish, baked server-side
//      by the SAME traced finish the viewer runs live, so it can never drift
//      from the sliders.
//   2. EVERY FINISH the pipeline lists — composites and single channels
//      alike — each with its build status, deep-linked into the viewer.
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
import PreviewImage from "./PreviewImage.jsx";


function FinishThumbnail({ objectType, objectId, pipeline, finish }) {
    const navigate = useNavigate();
    const hierarchy = [{ type: objectType, id: objectId }];
    const view = { pipeline, finish: finish.name };
    const built = finish.status !== 'absent';
    const url = built
        ? `${buildApiUrl('/tap/thumbnail', hierarchy, view)}?v=${finish.mtime}`
        : null;

    return (
        <Box
            onClick={() => navigate(buildUrl({ hierarchy, view }))}
            sx={{
                // Fill an equal share of the row's width: with N siblings each
                // `flex:1`, every image is 1/N of the viewport.
                flex: '1 1 0',
                minWidth: 0,
                aspectRatio: '3 / 2',
                maxHeight: '65vh',
                borderRadius: 1,
                border: '1px solid',
                borderColor: 'divider',
                overflow: 'hidden',
                position: 'relative',
                // Always clickable — even before the finish is built, the
                // viewer page is useful for the workflow graph + progress.
                cursor: 'pointer',
                backgroundColor: built ? 'transparent' : 'rgba(255,255,255,0.04)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
            }}
        >
            {url ? (
                <img
                    src={url}
                    alt={finish.name}
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
                {finish.name}
            </Box>
            {finish.status === 'out_of_date' && (
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


// Exported: TargetPanel renders the same per-pipeline sections — the
// component is object-type-agnostic (every URL/API it touches is keyed by
// (objectType, objectId)).
export function PipelineSection({ objectType, objectId, pipeline }) {
    const [discardOpen, setDiscardOpen] = useState(false);
    const navigate = useNavigate();
    const hierarchy = [{ type: objectType, id: objectId }];
    // THE DELIVERABLE: the first (preferred) finish that is actually built.
    const hero = pipeline.finishes.find((f) => f.status !== 'absent');
    const heroView = hero && { pipeline: pipeline.name, finish: hero.name };
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
            {heroView && (
                <Box
                    onClick={() => navigate(buildUrl({ hierarchy, view: heroView }))}
                    sx={{
                        cursor: 'pointer', mb: 2, position: 'relative',
                        alignSelf: 'flex-start', borderRadius: 1,
                        overflow: 'hidden', border: '1px solid',
                        borderColor: 'divider', maxWidth: '100%',
                    }}
                >
                    <PreviewImage
                        src={`${buildApiUrl('/tap/thumbnail', hierarchy, heroView)}?v=${hero.mtime}`}
                        alt={`${pipeline.name} — ${hero.name}`}
                        style={{ display: 'block', maxHeight: '48vh',
                                 maxWidth: '100%' }}
                    />
                </Box>
            )}

            {/* Every finish this pipeline lists, in preference order. The
                images share the full width (each flex:1). */}
            <Box sx={{
                display: 'flex', gap: 1, width: '100%',
                alignItems: 'flex-start',
            }}>
                {pipeline.finishes.map((f) => (
                    <FinishThumbnail
                        key={f.name}
                        objectType={objectType}
                        objectId={objectId}
                        pipeline={pipeline.name}
                        finish={f}
                    />
                ))}
            </Box>
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

    // No page-level hero: each PipelineSection renders its OWN canonical
    // preview, so a single auto-picked one on top would just duplicate
    // whichever pipeline happened to win — and on an A/B object it silently
    // implied the two pipelines shared a look.
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
