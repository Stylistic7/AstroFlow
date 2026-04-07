// MosaicViewer.jsx
import React, { useContext, useEffect, useMemo, useState, useCallback } from "react";
import { useParams, useSearchParams } from "react-router-dom";

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
    const [searchParams, setSearchParams] = useSearchParams();
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

	// Persist filter and quality level in URL query params so back-navigation restores them.
	const filter = searchParams.get("filter") || availableFilters[0];
	const setFilter = useCallback((f) => {
		setSearchParams((prev) => { prev.set("filter", f); return prev; }, { replace: true });
	}, [setSearchParams]);

	useEffect(() => {
		if (!availableFilters || availableFilters.length === 0) return;
		if (!availableFilters.includes(filter)) {
			setFilter(availableFilters[0]);
		}
	}, [mosaic_id, availableFilters, filter, setFilter]);

	const [qualityLevel, setQualityLevelState, enrichedLevels] =
		useQualityLevels("mosaic", mosaic_id, filter);

	// Sync quality level to/from URL query param.
	const qualityFromUrl = searchParams.get("quality");
	const setQualityLevel = useCallback((q) => {
		setQualityLevelState(q);
		setSearchParams((prev) => { prev.set("quality", q); return prev; }, { replace: true });
	}, [setQualityLevelState, setSearchParams]);

	useEffect(() => {
		if (qualityFromUrl && qualityFromUrl !== qualityLevel) {
			setQualityLevelState(qualityFromUrl);
		}
	}, [qualityFromUrl]);

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
