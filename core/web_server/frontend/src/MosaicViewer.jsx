// MosaicViewer.jsx
import React, { useContext, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

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

import AstroFlowContext from "./AstroFlowContext.jsx";
import { AstroFlowImage, useQualityLevels } from "./AstroFlowImage.jsx";
import ColorChannelSelector from "./ColorChannelSelector.jsx";
import QualityLevelSelector from "./QualityLevelSelector.jsx";
import TelescopiusDialog from "./TelescopiusDialog.jsx";

import ToggleIconButton from "./ToggleIconButton";
import WorkflowGraph from "./WorkflowGraph.jsx";
import darkTheme from "./theme.js";

export default function MosaicViewer() {
    const { mosaic_id } = useParams();
    const { stacks, targets, mosaics } = useContext(AstroFlowContext);

	const [tabIndex, setTabIndex] = useState(0);

	// View toggles (add more here later: indices, exposure_widgets, etc.)
	const [showOutline, setShowOutline] = useState(true);
    
    const mosaic = mosaics[mosaic_id];
    
	const availableFilters = useMemo(() => {
		if (!mosaic) return [];
		let filters = new Set();
		for (let target_id of mosaic.target_ids) {
			for (let stack_id of targets[target_id].stack_ids) {
				filters.add(stacks[stack_id].filter);
			}
		}
		return Array.from(filters);
	}, [mosaic, stacks, targets]);

	const [filter, setFilter] = useState(availableFilters[0]);

	useEffect(() => {
		if (!availableFilters || availableFilters.length === 0) return;
		setFilter((prev) =>
			availableFilters.includes(prev) ? prev : availableFilters[0]
		);
	}, [mosaic_id, availableFilters]);

	const [qualityLevel, setQualityLevel, enrichedLevels] =
		useQualityLevels("mosaic", mosaic_id, filter);

	const [telescopiusOpen, setTelescopiusOpen] = useState(false);

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

			{/* Control bar */}
			<Box
				sx={{
					display: "flex",
					alignItems: "center",
					justifyContent: "center",
					gap: 2,
					py: 1,
				}}
			>
				<QualityLevelSelector
					selected={qualityLevel}
					available={enrichedLevels}
					onSelect={setQualityLevel}
				/>
				<Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

				<ColorChannelSelector
					stacks={stacks}
					targets={targets}
					mosaics={mosaics}
					mosaic_id={mosaic_id}
					filter={filter}
					setFilter={setFilter}
				/>

				<Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

				<ToggleIconButton
					name="outline"
					icon={<FullscreenIcon />}
					checked={showOutline}
					handleChange={() => setShowOutline((v) => !v)}
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
						qualityLevel={qualityLevel}
						onQualityChange={setQualityLevel}
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
						url={`/api/workflow_graph/mosaic/${mosaic_id}/${filter}${qualityLevel ? `/${qualityLevel}` : ''}`}
					/>
				</Box>
			</Box>
		</Paper>
	);
}
