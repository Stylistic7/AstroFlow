// MosaicViewer.jsx — the detail page for a single
// (mosaic × implementation × quality × filter). The dashboard
// (MosaicDashboard.jsx) is the entry point; this is what you see
// after clicking a thumbnail.
import React, { useCallback, useContext, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

import {
	Box,
	Divider,
	IconButton,
	Paper,
	Tab,
	Tabs,
	Tooltip,
	Typography,
} from "@mui/material";

import FullscreenIcon from "@mui/icons-material/Fullscreen";
import ShareIcon from "@mui/icons-material/Share";
import PinIcon from "@mui/icons-material/Pin";
import LaunchIcon from "@mui/icons-material/Launch";

import AstroFlowContext from "./AstroFlowContext.jsx";
import { AstroFlowImage } from "./AstroFlowImage.jsx";
import TelescopiusDialog from "./TelescopiusDialog.jsx";
import ToggleIconButton from "./ToggleIconButton";
import WorkflowGraph from "./WorkflowGraph.jsx";
import ConcernsBadge from "./ConcernsBadge.jsx";
import darkTheme from "./theme.js";
import { useVersionsMatrix, useVersionCycling, versionExists } from "./useVersions.js";


export default function MosaicViewer() {
    const { mosaic_id, implementation, quality, filter } = useParams();
    const navigate = useNavigate();
    const { stacks, targets, mosaics, computeGraphStatus } = useContext(AstroFlowContext);

	const [tabIndex, setTabIndex] = useState(0);
	const [showOutline, setShowOutline] = useState(true);
	const [showPanelNumbers, setShowPanelNumbers] = useState(false);
	const [telescopiusOpen, setTelescopiusOpen] = useState(false);

    const mosaic = mosaics?.[mosaic_id];

    const versionsData = useVersionsMatrix("mosaic", mosaic_id, computeGraphStatus);

    // Keyboard shortcuts:  [ ]  implementation,  { }  quality,  , .  filter.
    const onCycle = useCallback(
        ({ implementation: i, quality: q, filter: f }) =>
            navigate(`/mosaic/${mosaic_id}/${i}/${q}/${f}`),
        [navigate, mosaic_id],
    );
    useVersionCycling({
        data: versionsData,
        current: { implementation, quality, filter },
        onChange: onCycle,
    });

    // The "Open in PixInsight" button needs to know whether the
    // canonical Master output (or its in-progress fallback) exists for
    // the current view; either path counts as available per the
    // versions matrix.
    const masterExists = versionExists(
        versionsData.versions, implementation, "Master", filter,
    );

    const openInPixInsight = () => {
        fetch(`/api/open_mosaic_master/${mosaic_id}/${implementation}/${filter}`)
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
				<Tooltip title="Export to Telescopius">
					<IconButton
						size="small"
						onClick={() => setTelescopiusOpen(true)}
						sx={{ color: 'text.secondary' }}
					>
						<ShareIcon fontSize="small" />
					</IconButton>
				</Tooltip>
				<Tooltip title={masterExists ? "Open in PixInsight" : "No master image available yet"}>
					<span>
						<IconButton
							size="small"
							disabled={!masterExists}
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
					quality={quality}
					implementation={implementation}
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
						qualityLevel={quality}
						implementation={implementation}
						showQualitySelector={false}
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
						url={`/api/workflow_graph/mosaic/${mosaic_id}/${filter}/${quality}`}
					/>
				</Box>
			</Box>
		</Paper>
	);
}
