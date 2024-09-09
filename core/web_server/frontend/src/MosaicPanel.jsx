import React, { useState, useRef, useEffect, useReducer, useContext } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";

import Typography from '@mui/material/Typography';
import DirectionsRunIcon from '@mui/icons-material/DirectionsRun';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import Snackbar from '@mui/material/Snackbar';
import { Checkbox, FormControlLabel, FormGroup } from '@mui/material';

import MosaicOutline from "./MosaicOutline.jsx";
import ColorChannelSelector from "./ColorChannelSelector.jsx";
import TargetPanel from "./TargetPanel.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';
import { ra_to_hms, dec_to_dms } from './units.js';
import DivWithActions from './DivWithActions.jsx';
import MosaicOutlineActionButtons from './MosaicOutlineActionButtons.jsx';
import ImageWithOverlay from "./ImageWithOverlay.jsx";

import "./Starmap.css";

function RuleOfThirdsOverlay({ width, height }) {
	return (
		<svg viewBox={`0 0 ${width} ${height}`}>
			<line x1="0" y1={height / 3} x2={width} y2={height / 3} stroke="white" strokeWidth="0.5" />
			<line x1="0" y1={(height / 3) * 2} x2={width} y2={(height / 3) * 2} stroke="white" strokeWidth="0.5" />
			<line x1={width / 3} y1="0" x2={width / 3} y2={height} stroke="white" strokeWidth="0.5" />
			<line x1={(width / 3) * 2} y1="0" x2={(width / 3) * 2} y2={height} stroke="white" strokeWidth="0.5" />
		</svg>
	);
}

// TODO: Add "mosaic not found" messages.
export default function MosaicPanel() {
	const { stacks, targets, mosaics } = useContext(AstroFlowContext);

	const params = useParams();
	const location = useLocation();
	const navigate = useNavigate();

	const mosaic = mosaics[params.mosaic_id];

	const [visibility, setVisibility] = useState({
		outline: true,
		indices: true,
		exposure_widgets: true,
	});

	const [scale, setScale] = useState(1.0)

	const handleChange = (name, checked) => {
		setVisibility({
			...visibility,
			[name]: checked,
		});
	};

	const get_filters = () => {
		let filters = new Set();
		for (let target_id of mosaic.target_ids) {
			for (let stack_id of targets[target_id].stack_ids) {
				filters.add(stacks[stack_id].filter);
			}
		}

		const filter_array = Array.from(filters);
		return filter_array;
	}

	const [filter, setFilter] = useState(get_filters()[0]);

	const fontColor = "#aaa";

	const handleSetFilter = f => {
		setFilter(f);
	};

	const telescopius_pane_line = (mosaic, target_id, pane) => {
		let [tile_x, tile_y] = mosaic.tile_indices[pane];
		let target = targets[target_id];
		let stack_id = target.stack_ids[0];
		let stack = stacks[stack_id];
		let [ra, dec] = stack.center;
		let rotation_angle = stack.rotation_angle * 180.0 / Math.PI;
		let pane_width = stack.image_scale * stack.width_pixels / 60.0;
		let pane_height = stack.image_scale * stack.height_pixels / 60.0;

		// TODO: The overlap percent is hardcoded here.
		return `Pane ${pane + 1}, ${ra_to_hms(ra)}, ${dec_to_dms(dec)}, ${rotation_angle.toFixed(1)}, ${pane_width.toFixed(2)}, ${pane_height.toFixed(2)}, 20%, ${tile_x + 1}, ${tile_y + 1}`;
	};

	const [open, setOpen] = useState(false);

	const telescopius = mosaic => {
		const header = "Pane, RA, DEC, Position Angle (East), Pane width (arcmins), Pane height (arcmins), Overlap, Row, Column";
		const text = [header].concat(
			mosaic.target_ids.map((target_id, pane) =>
				telescopius_pane_line(mosaic, target_id, pane)
			)).join('\n');

		return <>
			<p>
				{header}<br />
				{mosaic.target_ids.map((target_id, pane) => <>
					<span>
						{telescopius_pane_line(mosaic, target_id, pane)}
					</span><br />
				</>)}
			</p >
			<CopyToClipboard text={text}
				onCopy={() => setOpen(true)}>
				<button>Copy to clipboard with button</button>
			</CopyToClipboard>
		</>;
	};

	// TODO: Add other modes besides AUTO.
	const runAllTasks = () => {
		fetch(`/add_mosaic_to_shopping_cart/${params.mosaic_id}`);
	};

	const imageType = "mosaic_thumbnail_image_for_filter";

	const handleZoomIn = () => {
		setScale(scale * 1.2);
	}

	const handleZoomOut = () => {
		setScale(scale * 0.8);
	}

	const actionButtons = () => {
		return <MosaicOutlineActionButtons
			handleChange={handleChange}
			handleZoomIn={handleZoomIn}
			handleZoomOut={handleZoomOut}
		/>
	}

	const mosaicOutline = () => (props) => {
		return <MosaicOutline
			{...props}
			mosaic_id={params.mosaic_id}
			filter={filter}
			outline_visible={visibility.outline}
			indices_visible={visibility.indices}
			exposure_widgets_visible={visibility.exposure_widgets}
		/>;
	};

	// TODO: Add a dialog box to the "run" button that shows the user what will be
	// run, and how long it will likely take, and allow them to accept it.
	return (
		<>
			<Accordion>
				<AccordionSummary
					expandIcon={<ExpandMoreIcon />}
					aria-controls="telescopius-content"
					id="telescopius-header"
				>
					Telescopius
				</AccordionSummary>
				<AccordionDetails>
					{telescopius(mosaic)}
				</AccordionDetails>
			</Accordion>
			<Snackbar
				anchorOrigin={{ vertical: "top", horizontal: "center" }}
				open={open}
				autoHideDuration={5000}
				onClose={event => setOpen(false)}
				message="Telescopius text copied"
				key="telescopius-snackbar"
			/>

			<Typography color="text.primary" margin="0.5em" onClick={runAllTasks}>
				<DirectionsRunIcon sx={{ marginRight: '0.5em' }} onClick={runAllTasks} /> Run all tasks
			</Typography>

			<DivWithActions
				action_buttons={actionButtons()}
			>
				<div
					style={{
						display: "flex",
						flexDirection: "row",
						justifyContent: "center"
					}}>
					<ImageWithOverlay
						src={`/${imageType}/${params.mosaic_id}/${filter}`}
						alt="mosaic"
						scale={scale}
						overlay={mosaicOutline()}
					/>
				</div>
				<ColorChannelSelector
					stacks={stacks}
					targets={targets}
					mosaics={mosaics}
					mosaic_id={params.mosaic_id}
					filter={filter}
					setFilter={handleSetFilter}
				/>
			</DivWithActions>
		</>
	);
}
