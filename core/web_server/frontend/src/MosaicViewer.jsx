// MosaicViewer.jsx — the detail page for a single
// (mosaic × pipeline × finish). The dashboard (MosaicDashboard.jsx) is
// the entry point; this is what you see after clicking a thumbnail.
import React, { useContext, useState } from "react";
import { useParams } from "react-router-dom";

import {
	Box,
	Divider,
	IconButton,
	Paper,
	Snackbar,
	Tab,
	Tabs,
	Tooltip,
	Typography,
} from "@mui/material";

import FullscreenIcon from "@mui/icons-material/Fullscreen";
import ShareIcon from "@mui/icons-material/Share";
import PinIcon from "@mui/icons-material/Pin";
import LaunchIcon from "@mui/icons-material/Launch";
import MoreTimeIcon from "@mui/icons-material/MoreTime";

import AstroFlowContext from "./AstroFlowContext.jsx";
import { AstroFlowImage } from "./AstroFlowImage.jsx";
import TelescopiusDialog from "./TelescopiusDialog.jsx";
import ToggleIconButton from "./ToggleIconButton";
import HdrControls, { useHdrCapability } from "./HdrControls.jsx";
import WorkflowGraph from "./WorkflowGraph.jsx";
import ConcernsBadge from "./ConcernsBadge.jsx";
import darkTheme from "./theme.js";
import { useFinishes, useValidFinish } from "./useVersions.js";


export default function MosaicViewer() {
    // Serves BOTH /mosaic/:mosaic_id/... and /target/:target_id/... view
    // routes — a target's colour tail produces the same finishes a mosaic's
    // does, and every API here is keyed by (objectType, objectId). The
    // mosaic-only chrome (outline, panel numbers, depth, Telescopius) is
    // hidden on target pages.
    const { mosaic_id, target_id, pipeline, finish } = useParams();
    const objectType = target_id ? "target" : "mosaic";
    const objectId = target_id ?? mosaic_id;
    const { stacks, targets, mosaics, computeGraphStatus } = useContext(AstroFlowContext);

	const [tabIndex, setTabIndex] = useState(0);
	const [showOutline, setShowOutline] = useState(true);
	const [showPanelNumbers, setShowPanelNumbers] = useState(false);
	const [showDepth, setShowDepth] = useState(false);
	const [telescopiusOpen, setTelescopiusOpen] = useState(false);
	const hdrCapable = useHdrCapability();
	// Default ON where the display can actually show it. `useHdrCapability` is
	// a `(dynamic-range: high)` match, so this only self-enables on a screen
	// that has the headroom; an SDR display still sees peak forced to 1.0.
	const [hdrEnabled, setHdrEnabled] = useState(hdrCapable);
	const [peakBrightness, setPeakBrightness] = useState(2.0);

    const mosaic = mosaics?.[mosaic_id];
    const entity = target_id ? targets?.[target_id] : mosaic;

    const hierarchy = [];
    if (mosaic_id) hierarchy.push({ type: 'mosaic', id: mosaic_id });
    if (target_id) hierarchy.push({ type: 'target', id: target_id });

    const finishes = useFinishes(objectType, objectId, computeGraphStatus);
    useValidFinish(finishes, hierarchy, pipeline, finish);
    // A single-channel finish is named by its filter role; composites have
    // none. Concerns and the workflow graph are filed per role.
    const role = finishes?.pipelines.find((p) => p.name === pipeline)
        ?.finishes.find((f) => f.name === finish)?.role ?? null;

    const [openNote, setOpenNote] = useState(null);
    const openInPixInsight = async () => {
        // 202: the layers open in PixInsight as they land — a tiled
        // gigapixel layer is rewritten for it first (~25 s per channel).
        try {
            const r = await fetch(`/api/open/${objectType}/${objectId}/${encodeURIComponent(pipeline)}/${encodeURIComponent(finish)}`,
                                  { method: 'POST' });
            const d = await r.json();
            if (!r.ok) { setOpenNote(`Nothing to open: ${d.error || r.status}`); return; }
            const n = d.opening.length;
            setOpenNote(d.exporting.length
                ? `Preparing ${n} image${n > 1 ? 's' : ''} for PixInsight (${d.exporting.length} rewritten first, ~25 s each): ${d.opening.join(', ')}`
                : `Opening in PixInsight: ${d.opening.join(', ')}`);
        } catch (e) { console.error(e); setOpenNote("Open failed"); }
    };

    if (!entity) {
        return <Typography sx={{ p: 3 }}>Loading…</Typography>;
    }

	return (
		<Paper
			elevation={2}
			sx={{
				borderRadius: 0,
				borderTop: "1px solid",
				borderColor: "divider",
				display: "flex",
				flexDirection: "column",
				flex: 1,
				minHeight: 0,
			}}
		>
			{mosaic && <TelescopiusDialog
				open={telescopiusOpen}
				onClose={() => setTelescopiusOpen(false)}
				stacks={stacks}
				targets={targets}
				mosaic={mosaic}
			/>}

			{/* Pipeline / finish switching is handled by the pill crumbs in
			    AstroFlowBreadcrumbs. */}

			{/* Control bar */}
			<Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 2, py: 1 }}>
				{mosaic && <>
					<ToggleIconButton
						name="outline"
						icon={<FullscreenIcon />}
						checked={showOutline}
						handleChange={() => setShowOutline((v) => !v)}
					/>
					<ToggleIconButton
						name="panel numbers"
						icon={<PinIcon />}
						checked={showPanelNumbers}
						handleChange={() => setShowPanelNumbers((v) => !v)}
					/>
					<ToggleIconButton
						name="depth: hours to uniform"
						icon={<MoreTimeIcon />}
						checked={showDepth}
						handleChange={() => setShowDepth((v) => !v)}
					/>
				</>}
				{/* Peak brightness is the ONE global uniform: the finish shader
				    ends in `rgb * globals.peakBrightness`, whatever the finish.
				    `HdrControls` self-gates on real display capability. */}
				<HdrControls
					hdrCapable={hdrCapable}
					hdrEnabled={hdrEnabled}
					setHdrEnabled={setHdrEnabled}
					peakBrightness={peakBrightness}
					setPeakBrightness={setPeakBrightness}
				/>
				{mosaic && <Tooltip title="Export to Telescopius">
					<IconButton
						size="small"
						onClick={() => setTelescopiusOpen(true)}
						sx={{ color: 'text.secondary' }}
					>
						<ShareIcon fontSize="small" />
					</IconButton>
				</Tooltip>}
				<Tooltip title="Open this finish's inputs in PixInsight">
					<span>
						<IconButton
							size="small"
							onClick={openInPixInsight}
							sx={{ color: 'text.secondary' }}
						>
							<LaunchIcon fontSize="small" />
						</IconButton>
					</span>
				</Tooltip>
				<ConcernsBadge
					objectType={objectType}
					objectId={objectId}
					pipeline={pipeline}
					filter={role}
				/>
			</Box>

			{/* Tabs */}
			<Box sx={{ backgroundColor: "rgba(255,255,255,0.05)", px: 2 }}>
				<Tabs
					value={tabIndex}
					onChange={(_e, newIndex) => setTabIndex(newIndex)}
					indicatorColor="primary"
					textColor="primary"
					variant="scrollable"
					scrollButtons="auto"
				>
					<Tab label="Merged" />
					<Tab label="Graph" />
				</Tabs>
			</Box>
			<Divider />
			<Snackbar
				open={openNote !== null}
				autoHideDuration={8000}
				onClose={() => setOpenNote(null)}
				message={openNote || ""}
				anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
			/>

			{/* Tab content */}
			<Box sx={{ flex: 1, p: 3, minHeight: 0, position: "relative" }}>
				<Box
					hidden={tabIndex !== 0}
					sx={{
						height: "100%",
						overflow: "hidden",
						position: "absolute",
						inset: 0,
						p: 3,
					}}
				>
					<AstroFlowImage
						objectType={objectType}
						objectId={objectId}
						showPolygons={mosaic ? showOutline : false}
						showPanelNumbers={mosaic ? showPanelNumbers : false}
						showDepth={mosaic ? showDepth : false}
						pipeline={pipeline}
						finish={finish}
						hdrEnabled={hdrEnabled}
						peakBrightness={peakBrightness}
					/>
				</Box>

				<Box
					hidden={tabIndex !== 1}
					sx={{
						height: "100%",
						overflow: "auto",
						position: "absolute",
						inset: 0,
						p: 3,
					}}
				>
					<WorkflowGraph
						theme={darkTheme}
						url={
							`/api/workflow_graph/${objectType}/${objectId}`
							+ `/${encodeURIComponent(pipeline)}`
							+ (role ? `/${encodeURIComponent(role)}` : "")
						}
					/>
				</Box>
			</Box>
		</Paper>
	);
}
