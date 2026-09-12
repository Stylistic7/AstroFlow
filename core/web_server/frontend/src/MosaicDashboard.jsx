// MosaicDashboard.jsx — at-a-glance view of a mosaic: what each pipeline
// PRODUCED, and whether it is current. One section per registered pipeline,
// each showing three kinds of thing and nothing else:
//
//   1. THE DELIVERABLE — that pipeline's canonical finished preview, rendered
//      server-side by the SAME traced finish the viewer runs live, so it can
//      never drift from the sliders. Pinned with `?pipeline=`, because on an
//      A/B object an auto-picked preview renders one image under both headings.
//   2. THE SANITY CHECK — the raw per-filter channels (Mosaic / Merged /
//      Subframe / Master). One band, autostretched, is an honest picture of
//      that band: a bad panel or registration slip shows up immediately.
//   3. THE PLUMBING — the multi-band linear scene formats, as STATUS ROWS, not
//      pictures. See `isSceneFormat` for why rendering them was worse than
//      merely ugly.
//
// This replaced a uniform grid of one thumbnail per (tap × filter) — a design
// from when the pipeline BAKED Hubble/Foraxx images to disk. Once colour moved
// into the viewer's live WGSL finishes, every tile that grid showed became an
// intermediate, and the page was showing the plumbing.
//
// Data source: `/api/pipeline_outputs/mosaic/<id>` returns one entry per
// pipeline, each with its tap points (display labels, in trace order) and, per
// tap, the real (filter, on-disk-existence, status) outputs. Tap points are
// discovered at pipeline-trace time from `@makes_image(display=...)`
// annotations, so user pipelines surface here automatically — and classify
// themselves, since the split above is derived from the outputs, not a
// hardcoded list of display names.
import React, { useContext, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Box, Paper, Typography, Divider, Button } from "@mui/material";
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep";

import AstroFlowContext from "./AstroFlowContext.jsx";
import DiscardResultsDialog from "./DiscardResultsDialog.jsx";
import { buildUrl, buildApiUrl } from "./urls.js";
import { useVersionsMatrix } from "./useVersions.js";
import PreviewImage from "./PreviewImage.jsx";


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


// A tap whose outputs are ALL "RGB" is a multi-band LINEAR SCENE FORMAT —
// Linear SHO / Narrowband / RGB / RGB Stars — i.e. the viewer's INPUT BUFFER,
// not a picture. We deliberately don't render those as images (2026-08-16):
//
//   * they are ugly for a real reason — Linear SHO is S/H/O crammed into RGB
//     with no palette; the meaning arrives in the viewer's WGSL finish;
//   * worse, the thumbnail is autostretched by the THUMBNAIL path while the
//     viewer renders the same tap through a live finish, so the tile shows a
//     transfer function no product actually has. A sickly green tile then reads
//     as "bad data" when the data is fine — a false-alarm generator.
//
// Per-FILTER taps (Mosaic / Merged / Subframe / Master / Channel) stay as
// images: one band, autostretched, is an honest picture of that band and the
// cheapest sanity check there is for a bad panel or a registration slip.
// Deriving this from the OUTPUTS rather than a display-name list means user
// pipelines classify themselves.
function isSceneFormat(tap) {
    const outs = tap.outputs || [];
    return outs.length > 0 && outs.every((o) => String(o.filter) === "RGB");
}


// The scene formats, as a compact status line instead of a misleading picture:
// what exists, whether it's current, and a way into the viewer that renders it
// properly.
function SceneFormatRow({ objectType, objectId, pipeline, tap }) {
    const navigate = useNavigate();
    const o = (tap.outputs || [])[0] || {};
    const href = buildUrl({
        hierarchy: [{ type: objectType, id: objectId }],
        view: { pipeline, display: tap.display, filter: o.filter },
    });
    const tone = !o.exists ? 'text.disabled'
        : o.status === 'out_of_date' ? 'warning.main' : 'success.main';
    const label = !o.exists ? 'not built'
        : o.status === 'out_of_date' ? 'out of date' : 'built';
    return (
        <Box
            onClick={() => navigate(href)}
            sx={{
                display: 'flex', alignItems: 'center', gap: 1.5,
                px: 1.5, py: 0.75, borderRadius: 1, cursor: 'pointer',
                border: '1px solid', borderColor: 'divider',
                '&:hover': { borderColor: 'primary.main' },
            }}
        >
            <Typography variant="body2" sx={{ flexGrow: 1 }}>
                {tap.display}
            </Typography>
            <Typography variant="caption" sx={{ color: tone, fontWeight: 600 }}>
                {label}
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                open in viewer →
            </Typography>
        </Box>
    );
}


// Exported: TargetPanel renders the same per-pipeline tap sections — the
// component is object-type-agnostic (every URL/API it touches is keyed by
// (objectType, objectId)).
export function PipelineSection({ objectType, objectId, pipeline }) {
    const [discardOpen, setDiscardOpen] = useState(false);
    const navigate = useNavigate();
    const scene = (pipeline.taps || []).filter(isSceneFormat);
    const channels = (pipeline.taps || []).filter((t) => !isSceneFormat(t));
    // Deep-link the finished preview at the first built scene format of THIS
    // pipeline, so the click lands in the viewer on the pixels it just showed.
    const first = scene.find((t) => (t.outputs || []).some((o) => o.exists));
    const heroView = first && {
        pipeline: pipeline.name, display: first.display,
        filter: (first.outputs.find((o) => o.exists) || {}).filter,
    };
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
            {/* THE DELIVERABLE: this pipeline's canonical finished look,
                rendered server-side by the SAME traced finish the viewer runs,
                so it can never drift from what the sliders show. `?pipeline=`
                pins it — without that, an A/B renders one image twice. */}
            {heroView && (
                <Box
                    onClick={() => navigate(buildUrl({
                        hierarchy: [{ type: objectType, id: objectId }],
                        view: heroView,
                    }))}
                    sx={{
                        cursor: 'pointer', mb: 2, position: 'relative',
                        alignSelf: 'flex-start', borderRadius: 1,
                        overflow: 'hidden', border: '1px solid',
                        borderColor: 'divider', maxWidth: '100%',
                    }}
                >
                    <PreviewImage
                        src={`/thumbnail/preview/${objectType}/${objectId}`
                             + `?pipeline=${encodeURIComponent(pipeline.name)}`}
                        alt={`${pipeline.name} — canonical preview`}
                        style={{ display: 'block', maxHeight: '48vh',
                                 maxWidth: '100%' }}
                    />
                </Box>
            )}

            {/* THE SANITY CHECK: raw per-filter channels, one band each. */}
            {channels.map((tap) => (
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

            {/* THE PLUMBING: status only, never a picture. */}
            {scene.length > 0 && (
                <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" sx={{
                        color: 'text.secondary', mb: 0.5,
                    }}>
                        Scene formats
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        {scene.map((tap) => (
                            <SceneFormatRow
                                key={tap.display}
                                objectType={objectType}
                                objectId={objectId}
                                pipeline={pipeline.name}
                                tap={tap}
                            />
                        ))}
                    </Box>
                </Box>
            )}
        </Box>
    );
}


export default function MosaicDashboard() {
    const { mosaic_id } = useParams();
    const { mosaics, computeGraphStatus } = useContext(AstroFlowContext);
    const data = useVersionsMatrix("mosaic", mosaic_id, computeGraphStatus);

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
