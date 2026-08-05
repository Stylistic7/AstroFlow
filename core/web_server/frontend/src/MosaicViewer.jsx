// MosaicViewer.jsx — the detail page for a single
// (mosaic × implementation × quality × filter). The dashboard
// (MosaicDashboard.jsx) is the entry point; this is what you see
// after clicking a thumbnail.
import React, { useCallback, useContext, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

import {
	Box,
	Divider,
	IconButton,
	MenuItem,
	Paper,
	Select,
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
import { STRETCH_MODES } from "./tile_viewer/TileViewerV2.jsx";
import WorkflowGraph from "./WorkflowGraph.jsx";
import ConcernsBadge from "./ConcernsBadge.jsx";
import darkTheme from "./theme.js";
import { useVersionsMatrix } from "./useVersions.js";
import { buildUrl, buildApiUrl } from "./urls.js";


export default function MosaicViewer() {
    const { mosaic_id, pipeline, display, filter } = useParams();
    const navigate = useNavigate();
    const { stacks, targets, mosaics, computeGraphStatus } = useContext(AstroFlowContext);

	const [tabIndex, setTabIndex] = useState(0);
	const [showOutline, setShowOutline] = useState(true);
	const [showPanelNumbers, setShowPanelNumbers] = useState(false);
	const [showDepth, setShowDepth] = useState(false);
	const [telescopiusOpen, setTelescopiusOpen] = useState(false);
	const hdrCapable = useHdrCapability();
	const [hdrEnabled, setHdrEnabled] = useState(false);
	const [peakBrightness, setPeakBrightness] = useState(2.0);
	const [stretchMode, setStretchMode] = useState('stf');
	// True once the viewer reports the current tap is a baked-stretch
	// color image (>=3 bands): its stretch is baked in, so HDR + stretch
	// controls don't apply and are hidden.
	const [isColor, setIsColor] = useState(false);

    const mosaic = mosaics?.[mosaic_id];

    const versionsData = useVersionsMatrix("mosaic", mosaic_id, computeGraphStatus);

    // The URL might land on a (pipeline, display) combo this mosaic
    // doesn't actually have — typically because the user switched
    // pipelines on an older build of the breadcrumb (which used to
    // carry the current display across pipelines blindly). Redirect
    // to a valid (pipeline, display) for THIS pipeline so the page
    // doesn't 404-loop on every tile-metadata fetch.
    useEffect(() => {
        if (!pipeline || !display) return;
        const pipelines = versionsData?.pipelines || [];  // null until first fetch
        if (pipelines.length === 0) return;
        const current = pipelines.find((p) => p.name === pipeline);
        if (!current) return;
        const validDisplays = (current.taps || []).map((t) => t.display);
        if (validDisplays.includes(display)) return;
        const replacement = validDisplays[0];
        if (!replacement) return;
        navigate(buildUrl({
            hierarchy: [{ type: 'mosaic', id: mosaic_id }],
            view: { pipeline, display: replacement, filter },
        }), { replace: true });
    }, [pipeline, display, filter, mosaic_id, versionsData, navigate]);

    const openInPixInsight = () => {
        // open_mosaic_master's hierarchy is just <mosaic_id> (not
        // <type>/<id>), so spell out the prefix explicitly.
        const tap = `${encodeURIComponent(pipeline)}/${encodeURIComponent(display)}/${encodeURIComponent(filter)}`;
        fetch(`/api/open_mosaic_master/${mosaic_id}/${tap}`)
            .then((r) => { if (!r.ok) console.warn("open_mosaic_master failed", r.status); })
            .catch(console.error);
    };

    if (!mosaic) {
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
			<TelescopiusDialog
				open={telescopiusOpen}
				onClose={() => setTelescopiusOpen(false)}
				stacks={stacks}
				targets={targets}
				mosaic={mosaic}
			/>

			{/* Implementation / quality / filter switching is now handled
			    by the pill crumbs in AstroFlowBreadcrumbs. */}

			{/* Control bar */}
			<Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 2, py: 1 }}>
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
				{!isColor && (<>
					<HdrControls
						hdrCapable={hdrCapable}
						hdrEnabled={hdrEnabled}
						setHdrEnabled={setHdrEnabled}
						peakBrightness={peakBrightness}
						setPeakBrightness={setPeakBrightness}
					/>
					<Tooltip title="Stretch mode">
						<Select
							size="small"
							value={stretchMode}
							onChange={(e) => setStretchMode(e.target.value)}
							sx={{
								minWidth: 160,
								'& .MuiSelect-select': { py: 0.5, fontSize: '0.85rem' },
							}}
						>
							{Object.entries(STRETCH_MODES).map(([k, m]) => (
								<MenuItem key={k} value={k}>{m.label}</MenuItem>
							))}
						</Select>
					</Tooltip>
				</>)}
				<Tooltip title="Export to Telescopius">
					<IconButton
						size="small"
						onClick={() => setTelescopiusOpen(true)}
						sx={{ color: 'text.secondary' }}
					>
						<ShareIcon fontSize="small" />
					</IconButton>
				</Tooltip>
				<Tooltip title="Open in PixInsight">
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
					objectType="mosaic"
					objectId={mosaic_id}
					pipeline={pipeline}
					display={display}
					filter={filter}
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
						objectType="mosaic"
						objectId={mosaic_id}
						filter={filter}
						showPolygons={showOutline}
						showPanelNumbers={showPanelNumbers}
						showDepth={showDepth}
						pipeline={pipeline}
						display={display}
						showQualitySelector={false}
						hdrEnabled={hdrEnabled}
						peakBrightness={peakBrightness}
						stretchMode={stretchMode}
						onColorDetected={setIsColor}
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
							`/api/workflow_graph/mosaic/${mosaic_id}`
							+ `/${encodeURIComponent(pipeline)}`
							+ (filter ? `/${encodeURIComponent(filter)}` : "")
						}
					/>
				</Box>
			</Box>
		</Paper>
	);
}
