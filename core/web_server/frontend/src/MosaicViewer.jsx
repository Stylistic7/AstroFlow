// MosaicViewer.jsx
import React, { useContext, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import {
	Accordion,
	AccordionDetails,
	AccordionSummary,
	Box,
	Divider,
	Paper,
	Snackbar,
	Tab,
	Tabs,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import Typography from '@mui/material/Typography';

import FullscreenIcon from "@mui/icons-material/Fullscreen";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { createTheme } from "@mui/material/styles";

import AstroFlowContext from "./AstroFlowContext.jsx";
import { AstroFlowImage } from "./AstroFlowImage.jsx";
import ColorChannelSelector from "./ColorChannelSelector.jsx";
import ToggleIconButton from "./ToggleIconButton";
import WorkflowGraph from "./WorkflowGraph.jsx";
import { ra_to_hms, dec_to_dms } from "./units.js";

const darkTheme = createTheme({
	palette: {
		mode: "dark",
		primary: { main: "#cccccc" },
	},
});

// NOTE: This is the same helper logic you had previously, consolidated here.
const telescopius_pane_line = (stacks, targets, mosaic, target_id, pane) => {
	// TODO: Fix tile indices. Placeholder kept to preserve prior behavior.
	let tile_x = 1;
	let tile_y = 1;

	let target = targets[target_id];
	let stack_id = target.stack_ids[0];
	let stack = stacks[stack_id];

	let [ra, dec] = stack.center;
	let rotation_angle = (stack.rotation_angle * 180.0) / Math.PI;
	let pane_width = (stack.image_scale * stack.width_pixels) / 60.0;
	let pane_height = (stack.image_scale * stack.height_pixels) / 60.0;

	// TODO: Overlap percent was previously hardcoded.
	return `Pane ${pane + 1}, ${ra_to_hms(ra)}, ${dec_to_dms(
		dec
	)}, ${rotation_angle.toFixed(1)}, ${pane_width.toFixed(
		2
	)}, ${pane_height.toFixed(2)}, 20%, ${tile_x + 1}, ${tile_y + 1}`;
};

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

	const telescopius = useMemo(() => {
		if (!mosaic) return null;
		const header =
			"Pane, RA, DEC, Position Angle (East), Pane width (arcmins), Pane height (arcmins), Overlap, Row, Column";
		const lines = mosaic.target_ids.map((target_id, pane) =>
			telescopius_pane_line(stacks, targets, mosaic, target_id, pane)
		);
		const text = [header, ...lines].join("\n");
		return { header, lines, text };
	}, [mosaic, stacks, targets]);

	const [copiedOpen, setCopiedOpen] = useState(false);

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
			{/* Telescopius accordion moved above tabs */}
			<Accordion>
				<AccordionSummary
					expandIcon={<ExpandMoreIcon />}
					aria-controls="telescopius-content"
					id="telescopius-header"
				>
					Telescopius
				</AccordionSummary>
				<AccordionDetails>
					{telescopius ? (
						<>
							<p>
								{telescopius.header}
								<br />
								{telescopius.lines.map((line, idx) => (
									<React.Fragment key={idx}>
										<span>{line}</span>
										<br />
									</React.Fragment>
								))}
							</p>
							<CopyToClipboard
								text={telescopius.text}
								onCopy={() => setCopiedOpen(true)}
							>
								<button>Copy to clipboard with button</button>
							</CopyToClipboard>
						</>
					) : null}
				</AccordionDetails>
			</Accordion>

			<Snackbar
				anchorOrigin={{ vertical: "top", horizontal: "center" }}
				open={copiedOpen}
				autoHideDuration={5000}
				onClose={() => setCopiedOpen(false)}
				message="Telescopius text copied"
				key="telescopius-snackbar"
			/>

			{/* Control bar (ColorChannelSelector + view toggles) moved above tabs */}
			<Box
				sx={{
					display: "flex",
					alignItems: "center",
					justifyContent: "center",
					gap: 2,
					py: 1,
				}}
			>
				<ColorChannelSelector
					stacks={stacks}
					targets={targets}
					mosaics={mosaics}
					mosaic_id={mosaic_id}
					filter={filter}
					setFilter={setFilter}
				/>

				<Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

				<Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
					<ToggleIconButton
						name="outline"
						icon={<FullscreenIcon />}
						checked={showOutline}
						handleChange={() => setShowOutline((v) => !v)}
					/>
				</Box>
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
						url={`/workflow_graph/mosaic/${mosaic_id}/${filter}`}
					/>
				</Box>
			</Box>
		</Paper>
	);
}
